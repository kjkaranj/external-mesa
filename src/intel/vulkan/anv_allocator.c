/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <assert.h>
#include <linux/futex.h>
#include <linux/memfd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include "anv_private.h"

#include "util/hash_table.h"

#ifdef HAVE_VALGRIND
#define VG_NOACCESS_READ(__ptr) ({                       \
   VALGRIND_MAKE_MEM_DEFINED((__ptr), sizeof(*(__ptr))); \
   __typeof(*(__ptr)) __val = *(__ptr);                  \
   VALGRIND_MAKE_MEM_NOACCESS((__ptr), sizeof(*(__ptr)));\
   __val;                                                \
})
#define VG_NOACCESS_WRITE(__ptr, __val) ({                  \
   VALGRIND_MAKE_MEM_UNDEFINED((__ptr), sizeof(*(__ptr)));  \
   *(__ptr) = (__val);                                      \
   VALGRIND_MAKE_MEM_NOACCESS((__ptr), sizeof(*(__ptr)));   \
})
#else
#define VG_NOACCESS_READ(__ptr) (*(__ptr))
#define VG_NOACCESS_WRITE(__ptr, __val) (*(__ptr) = (__val))
#endif

/* Design goals:
 *
 *  - Lock free (except when resizing underlying bos)
 *
 *  - Constant time allocation with typically only one atomic
 *
 *  - Multiple allocation sizes without fragmentation
 *
 *  - Can grow while keeping addresses and offset of contents stable
 *
 *  - All allocations within one bo so we can point one of the
 *    STATE_BASE_ADDRESS pointers at it.
 *
 * The overall design is a two-level allocator: top level is a fixed size, big
 * block (8k) allocator, which operates out of a bo.  Allocation is done by
 * either pulling a block from the free list or growing the used range of the
 * bo.  Growing the range may run out of space in the bo which we then need to
 * grow.  Growing the bo is tricky in a multi-threaded, lockless environment:
 * we need to keep all pointers and contents in the old map valid.  GEM bos in
 * general can't grow, but we use a trick: we create a memfd and use ftruncate
 * to grow it as necessary.  We mmap the new size and then create a gem bo for
 * it using the new gem userptr ioctl.  Without heavy-handed locking around
 * our allocation fast-path, there isn't really a way to munmap the old mmap,
 * so we just keep it around until garbage collection time.  While the block
 * allocator is lockless for normal operations, we block other threads trying
 * to allocate while we're growing the map.  It sholdn't happen often, and
 * growing is fast anyway.
 *
 * At the next level we can use various sub-allocators.  The state pool is a
 * pool of smaller, fixed size objects, which operates much like the block
 * pool.  It uses a free list for freeing objects, but when it runs out of
 * space it just allocates a new block from the block pool.  This allocator is
 * intended for longer lived state objects such as SURFACE_STATE and most
 * other persistent state objects in the API.  We may need to track more info
 * with these object and a pointer back to the CPU object (eg VkImage).  In
 * those cases we just allocate a slightly bigger object and put the extra
 * state after the GPU state object.
 *
 * The state stream allocator works similar to how the i965 DRI driver streams
 * all its state.  Even with Vulkan, we need to emit transient state (whether
 * surface state base or dynamic state base), and for that we can just get a
 * block and fill it up.  These cases are local to a command buffer and the
 * sub-allocator need not be thread safe.  The streaming allocator gets a new
 * block when it runs out of space and chains them together so they can be
 * easily freed.
 */

/* Allocations are always at least 64 byte aligned, so 1 is an invalid value.
 * We use it to indicate the free list is empty. */
#define EMPTY 1

struct anv_mmap_cleanup {
   void *map;
   size_t size;
   uint32_t gem_handle;
};

#define ANV_MMAP_CLEANUP_INIT ((struct anv_mmap_cleanup){0})

static inline long
sys_futex(void *addr1, int op, int val1,
          struct timespec *timeout, void *addr2, int val3)
{
   return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}

static inline int
futex_wake(uint32_t *addr, int count)
{
   return sys_futex(addr, FUTEX_WAKE, count, NULL, NULL, 0);
}

static inline int
futex_wait(uint32_t *addr, int32_t value)
{
   return sys_futex(addr, FUTEX_WAIT, value, NULL, NULL, 0);
}

static inline int
memfd_create(const char *name, unsigned int flags)
{
   return syscall(SYS_memfd_create, name, flags);
}

static inline uint32_t
ilog2_round_up(uint32_t value)
{
   assert(value != 0);
   return 32 - __builtin_clz(value - 1);
}

static inline uint32_t
round_to_power_of_two(uint32_t value)
{
   return 1 << ilog2_round_up(value);
}

static bool
anv_free_list_pop(union anv_free_list *list, void **map, int32_t *offset)
{
   union anv_free_list current, new, old;

   current.u64 = list->u64;
   while (current.offset != EMPTY) {
      /* We have to add a memory barrier here so that the list head (and
       * offset) gets read before we read the map pointer.  This way we
       * know that the map pointer is valid for the given offset at the
       * point where we read it.
       */
      __sync_synchronize();

      int32_t *next_ptr = *map + current.offset;
      new.offset = VG_NOACCESS_READ(next_ptr);
      new.count = current.count + 1;
      old.u64 = __sync_val_compare_and_swap(&list->u64, current.u64, new.u64);
      if (old.u64 == current.u64) {
         *offset = current.offset;
         return true;
      }
      current = old;
   }

   return false;
}

static void
anv_free_list_push(union anv_free_list *list, void *map, int32_t offset)
{
   union anv_free_list current, old, new;
   int32_t *next_ptr = map + offset;

   old = *list;
   do {
      current = old;
      VG_NOACCESS_WRITE(next_ptr, current.offset);
      new.offset = offset;
      new.count = current.count + 1;
      old.u64 = __sync_val_compare_and_swap(&list->u64, current.u64, new.u64);
   } while (old.u64 != current.u64);
}

/* All pointers in the ptr_free_list are assumed to be page-aligned.  This
 * means that the bottom 12 bits should all be zero.
 */
#define PFL_COUNT(x) ((uintptr_t)(x) & 0xfff)
#define PFL_PTR(x) ((void *)((uintptr_t)(x) & ~(uintptr_t)0xfff))
#define PFL_PACK(ptr, count) ({           \
   (void *)(((uintptr_t)(ptr) & ~(uintptr_t)0xfff) | ((count) & 0xfff)); \
})

static bool
anv_ptr_free_list_pop(void **list, void **elem)
{
   void *current = *list;
   while (PFL_PTR(current) != NULL) {
      void **next_ptr = PFL_PTR(current);
      void *new_ptr = VG_NOACCESS_READ(next_ptr);
      unsigned new_count = PFL_COUNT(current) + 1;
      void *new = PFL_PACK(new_ptr, new_count);
      void *old = __sync_val_compare_and_swap(list, current, new);
      if (old == current) {
         *elem = PFL_PTR(current);
         return true;
      }
      current = old;
   }

   return false;
}

static void
anv_ptr_free_list_push(void **list, void *elem)
{
   void *old, *current;
   void **next_ptr = elem;

   /* The pointer-based free list requires that the pointer be
    * page-aligned.  This is because we use the bottom 12 bits of the
    * pointer to store a counter to solve the ABA concurrency problem.
    */
   assert(((uintptr_t)elem & 0xfff) == 0);

   old = *list;
   do {
      current = old;
      VG_NOACCESS_WRITE(next_ptr, PFL_PTR(current));
      unsigned new_count = PFL_COUNT(current) + 1;
      void *new = PFL_PACK(elem, new_count);
      old = __sync_val_compare_and_swap(list, current, new);
   } while (old != current);
}

static uint32_t
anv_block_pool_grow(struct anv_block_pool *pool, struct anv_block_state *state);

VkResult
anv_block_pool_init(struct anv_block_pool *pool,
                    struct anv_device *device, uint32_t block_size)
{
   VkResult result;

   assert(util_is_power_of_two(block_size));

   pool->device = device;
   anv_bo_init(&pool->bo, 0, 0);
   pool->block_size = block_size;
   pool->free_list = ANV_FREE_LIST_EMPTY;
   pool->back_free_list = ANV_FREE_LIST_EMPTY;

   pool->fd = memfd_create("block pool", MFD_CLOEXEC);
   if (pool->fd == -1)
      return vk_error(VK_ERROR_INITIALIZATION_FAILED);

   /* Just make it 2GB up-front.  The Linux kernel won't actually back it
    * with pages until we either map and fault on one of them or we use
    * userptr and send a chunk of it off to the GPU.
    */
   if (ftruncate(pool->fd, BLOCK_POOL_MEMFD_SIZE) == -1) {
      result = vk_error(VK_ERROR_INITIALIZATION_FAILED);
      goto fail_fd;
   }

   if (!u_vector_init(&pool->mmap_cleanups,
                      round_to_power_of_two(sizeof(struct anv_mmap_cleanup)),
                      128)) {
      result = vk_error(VK_ERROR_INITIALIZATION_FAILED);
      goto fail_fd;
   }

   pool->state.next = 0;
   pool->state.end = 0;
   pool->back_state.next = 0;
   pool->back_state.end = 0;

   /* Immediately grow the pool so we'll have a backing bo. */
   pool->state.end = anv_block_pool_grow(pool, &pool->state);

   return VK_SUCCESS;

 fail_fd:
   close(pool->fd);

   return result;
}

void
anv_block_pool_finish(struct anv_block_pool *pool)
{
   struct anv_mmap_cleanup *cleanup;

   u_vector_foreach(cleanup, &pool->mmap_cleanups) {
      if (cleanup->map)
         munmap(cleanup->map, cleanup->size);
      if (cleanup->gem_handle)
         anv_gem_close(pool->device, cleanup->gem_handle);
   }

   u_vector_finish(&pool->mmap_cleanups);

   close(pool->fd);
}

#define PAGE_SIZE 4096

/** Grows and re-centers the block pool.
 *
 * We grow the block pool in one or both directions in such a way that the
 * following conditions are met:
 *
 *  1) The size of the entire pool is always a power of two.
 *
 *  2) The pool only grows on both ends.  Neither end can get
 *     shortened.
 *
 *  3) At the end of the allocation, we have about twice as much space
 *     allocated for each end as we have used.  This way the pool doesn't
 *     grow too far in one direction or the other.
 *
 *  4) If the _alloc_back() has never been called, then the back portion of
 *     the pool retains a size of zero.  (This makes it easier for users of
 *     the block pool that only want a one-sided pool.)
 *
 *  5) We have enough space allocated for at least one more block in
 *     whichever side `state` points to.
 *
 *  6) The center of the pool is always aligned to both the block_size of
 *     the pool and a 4K CPU page.
 */
static uint32_t
anv_block_pool_grow(struct anv_block_pool *pool, struct anv_block_state *state)
{
   size_t size;
   void *map;
   uint32_t gem_handle;
   struct anv_mmap_cleanup *cleanup;

   pthread_mutex_lock(&pool->device->mutex);

   assert(state == &pool->state || state == &pool->back_state);

   /* Gather a little usage information on the pool.  Since we may have
    * threadsd waiting in queue to get some storage while we resize, it's
    * actually possible that total_used will be larger than old_size.  In
    * particular, block_pool_alloc() increments state->next prior to
    * calling block_pool_grow, so this ensures that we get enough space for
    * which ever side tries to grow the pool.
    *
    * We align to a page size because it makes it easier to do our
    * calculations later in such a way that we state page-aigned.
    */
   uint32_t back_used = align_u32(pool->back_state.next, PAGE_SIZE);
   uint32_t front_used = align_u32(pool->state.next, PAGE_SIZE);
   uint32_t total_used = front_used + back_used;

   assert(state == &pool->state || back_used > 0);

   size_t old_size = pool->bo.size;

   if (old_size != 0 &&
       back_used * 2 <= pool->center_bo_offset &&
       front_used * 2 <= (old_size - pool->center_bo_offset)) {
      /* If we're in this case then this isn't the firsta allocation and we
       * already have enough space on both sides to hold double what we
       * have allocated.  There's nothing for us to do.
       */
      goto done;
   }

   if (old_size == 0) {
      /* This is the first allocation */
      size = MAX2(32 * pool->block_size, PAGE_SIZE);
   } else {
      size = old_size * 2;
   }

   /* We can't have a block pool bigger than 1GB because we use signed
    * 32-bit offsets in the free list and we don't want overflow.  We
    * should never need a block pool bigger than 1GB anyway.
    */
   assert(size <= (1u << 31));

   /* We compute a new center_bo_offset such that, when we double the size
    * of the pool, we maintain the ratio of how much is used by each side.
    * This way things should remain more-or-less balanced.
    */
   uint32_t center_bo_offset;
   if (back_used == 0) {
      /* If we're in this case then we have never called alloc_back().  In
       * this case, we want keep the offset at 0 to make things as simple
       * as possible for users that don't care about back allocations.
       */
      center_bo_offset = 0;
   } else {
      /* Try to "center" the allocation based on how much is currently in
       * use on each side of the center line.
       */
      center_bo_offset = ((uint64_t)size * back_used) / total_used;

      /* Align down to a multiple of both the block size and page size */
      uint32_t granularity = MAX2(pool->block_size, PAGE_SIZE);
      assert(util_is_power_of_two(granularity));
      center_bo_offset &= ~(granularity - 1);

      assert(center_bo_offset >= back_used);

      /* Make sure we don't shrink the back end of the pool */
      if (center_bo_offset < pool->back_state.end)
         center_bo_offset = pool->back_state.end;

      /* Make sure that we don't shrink the front end of the pool */
      if (size - center_bo_offset < pool->state.end)
         center_bo_offset = size - pool->state.end;
   }

   assert(center_bo_offset % pool->block_size == 0);
   assert(center_bo_offset % PAGE_SIZE == 0);

   /* Assert that we only ever grow the pool */
   assert(center_bo_offset >= pool->back_state.end);
   assert(size - center_bo_offset >= pool->state.end);

   cleanup = u_vector_add(&pool->mmap_cleanups);
   if (!cleanup)
      goto fail;
   *cleanup = ANV_MMAP_CLEANUP_INIT;

   /* Just leak the old map until we destroy the pool.  We can't munmap it
    * without races or imposing locking on the block allocate fast path. On
    * the whole the leaked maps adds up to less than the size of the
    * current map.  MAP_POPULATE seems like the right thing to do, but we
    * should try to get some numbers.
    */
   map = mmap(NULL, size, PROT_READ | PROT_WRITE,
              MAP_SHARED | MAP_POPULATE, pool->fd,
              BLOCK_POOL_MEMFD_CENTER - center_bo_offset);
   cleanup->map = map;
   cleanup->size = size;

   if (map == MAP_FAILED)
      goto fail;

   gem_handle = anv_gem_userptr(pool->device, map, size);
   if (gem_handle == 0)
      goto fail;
   cleanup->gem_handle = gem_handle;

#if 0
   /* Regular objects are created I915_CACHING_CACHED on LLC platforms and
    * I915_CACHING_NONE on non-LLC platforms. However, userptr objects are
    * always created as I915_CACHING_CACHED, which on non-LLC means
    * snooped. That can be useful but comes with a bit of overheard.  Since
    * we're eplicitly clflushing and don't want the overhead we need to turn
    * it off. */
   if (!pool->device->info.has_llc) {
      anv_gem_set_caching(pool->device, gem_handle, I915_CACHING_NONE);
      anv_gem_set_domain(pool->device, gem_handle,
                         I915_GEM_DOMAIN_GTT, I915_GEM_DOMAIN_GTT);
   }
#endif

   /* Now that we successfull allocated everything, we can write the new
    * values back into pool. */
   pool->map = map + center_bo_offset;
   pool->center_bo_offset = center_bo_offset;

   /* For block pool BOs we have to be a bit careful about where we place them
    * in the GTT.  There are two documented workarounds for state base address
    * placement : Wa32bitGeneralStateOffset and Wa32bitInstructionBaseOffset
    * which state that those two base addresses do not support 48-bit
    * addresses and need to be placed in the bottom 32-bit range.
    * Unfortunately, this is not quite accurate.
    *
    * The real problem is that we always set the size of our state pools in
    * STATE_BASE_ADDRESS to 0xfffff (the maximum) even though the BO is most
    * likely significantly smaller.  We do this because we do not no at the
    * time we emit STATE_BASE_ADDRESS whether or not we will need to expand
    * the pool during command buffer building so we don't actually have a
    * valid final size.  If the address + size, as seen by STATE_BASE_ADDRESS
    * overflows 48 bits, the GPU appears to treat all accesses to the buffer
    * as being out of bounds and returns zero.  For dynamic state, this
    * usually just leads to rendering corruptions, but shaders that are all
    * zero hang the GPU immediately.
    *
    * The easiest solution to do is exactly what the bogus workarounds say to
    * do: restrict these buffers to 32-bit addresses.  We could also pin the
    * BO to some particular location of our choosing, but that's significantly
    * more work than just not setting a flag.  So, we explicitly DO NOT set
    * the EXEC_OBJECT_SUPPORTS_48B_ADDRESS flag and the kernel does all of the
    * hard work for us.
    */
   anv_bo_init(&pool->bo, gem_handle, size);
   pool->bo.map = map;

   if (pool->device->instance->physicalDevice.has_exec_async)
      pool->bo.flags |= EXEC_OBJECT_ASYNC;

done:
   pthread_mutex_unlock(&pool->device->mutex);

   /* Return the appropreate new size.  This function never actually
    * updates state->next.  Instead, we let the caller do that because it
    * needs to do so in order to maintain its concurrency model.
    */
   if (state == &pool->state) {
      return pool->bo.size - pool->center_bo_offset;
   } else {
      assert(pool->center_bo_offset > 0);
      return pool->center_bo_offset;
   }

fail:
   pthread_mutex_unlock(&pool->device->mutex);

   return 0;
}

static uint32_t
anv_block_pool_alloc_new(struct anv_block_pool *pool,
                         struct anv_block_state *pool_state)
{
   struct anv_block_state state, old, new;

   while (1) {
      state.u64 = __sync_fetch_and_add(&pool_state->u64, pool->block_size);
      if (state.next < state.end) {
         assert(pool->map);
         return state.next;
      } else if (state.next == state.end) {
         /* We allocated the first block outside the pool, we have to grow it.
          * pool_state->next acts a mutex: threads who try to allocate now will
          * get block indexes above the current limit and hit futex_wait
          * below. */
         new.next = state.next + pool->block_size;
         new.end = anv_block_pool_grow(pool, pool_state);
         assert(new.end >= new.next && new.end % pool->block_size == 0);
         old.u64 = __sync_lock_test_and_set(&pool_state->u64, new.u64);
         if (old.next != state.next)
            futex_wake(&pool_state->end, INT_MAX);
         return state.next;
      } else {
         futex_wait(&pool_state->end, state.end);
         continue;
      }
   }
}

int32_t
anv_block_pool_alloc(struct anv_block_pool *pool)
{
   int32_t offset;

   /* Try free list first. */
   if (anv_free_list_pop(&pool->free_list, &pool->map, &offset)) {
      assert(offset >= 0);
      assert(pool->map);
      return offset;
   }

   return anv_block_pool_alloc_new(pool, &pool->state);
}

/* Allocates a block out of the back of the block pool.
 *
 * This will allocated a block earlier than the "start" of the block pool.
 * The offsets returned from this function will be negative but will still
 * be correct relative to the block pool's map pointer.
 *
 * If you ever use anv_block_pool_alloc_back, then you will have to do
 * gymnastics with the block pool's BO when doing relocations.
 */
int32_t
anv_block_pool_alloc_back(struct anv_block_pool *pool)
{
   int32_t offset;

   /* Try free list first. */
   if (anv_free_list_pop(&pool->back_free_list, &pool->map, &offset)) {
      assert(offset < 0);
      assert(pool->map);
      return offset;
   }

   offset = anv_block_pool_alloc_new(pool, &pool->back_state);

   /* The offset we get out of anv_block_pool_alloc_new() is actually the
    * number of bytes downwards from the middle to the end of the block.
    * We need to turn it into a (negative) offset from the middle to the
    * start of the block.
    */
   assert(offset >= 0);
   return -(offset + pool->block_size);
}

void
anv_block_pool_free(struct anv_block_pool *pool, int32_t offset)
{
   if (offset < 0) {
      anv_free_list_push(&pool->back_free_list, pool->map, offset);
   } else {
      anv_free_list_push(&pool->free_list, pool->map, offset);
   }
}

static void
anv_fixed_size_state_pool_init(struct anv_fixed_size_state_pool *pool,
                               size_t state_size)
{
   /* At least a cache line and must divide the block size. */
   assert(state_size >= 64 && util_is_power_of_two(state_size));

   pool->state_size = state_size;
   pool->free_list = ANV_FREE_LIST_EMPTY;
   pool->block.next = 0;
   pool->block.end = 0;
}

static uint32_t
anv_fixed_size_state_pool_alloc(struct anv_fixed_size_state_pool *pool,
                                struct anv_block_pool *block_pool)
{
   int32_t offset;
   struct anv_block_state block, old, new;

   /* Try free list first. */
   if (anv_free_list_pop(&pool->free_list, &block_pool->map, &offset)) {
      assert(offset >= 0);
      return offset;
   }

   /* If free list was empty (or somebody raced us and took the items) we
    * allocate a new item from the end of the block */
 restart:
   block.u64 = __sync_fetch_and_add(&pool->block.u64, pool->state_size);

   if (block.next < block.end) {
      return block.next;
   } else if (block.next == block.end) {
      offset = anv_block_pool_alloc(block_pool);
      new.next = offset + pool->state_size;
      new.end = offset + block_pool->block_size;
      old.u64 = __sync_lock_test_and_set(&pool->block.u64, new.u64);
      if (old.next != block.next)
         futex_wake(&pool->block.end, INT_MAX);
      return offset;
   } else {
      futex_wait(&pool->block.end, block.end);
      goto restart;
   }
}

static void
anv_fixed_size_state_pool_free(struct anv_fixed_size_state_pool *pool,
                               struct anv_block_pool *block_pool,
                               uint32_t offset)
{
   anv_free_list_push(&pool->free_list, block_pool->map, offset);
}

void
anv_state_pool_init(struct anv_state_pool *pool,
                    struct anv_block_pool *block_pool)
{
   pool->block_pool = block_pool;
   for (unsigned i = 0; i < ANV_STATE_BUCKETS; i++) {
      size_t size = 1 << (ANV_MIN_STATE_SIZE_LOG2 + i);
      anv_fixed_size_state_pool_init(&pool->buckets[i], size);
   }
   VG(VALGRIND_CREATE_MEMPOOL(pool, 0, false));
}

void
anv_state_pool_finish(struct anv_state_pool *pool)
{
   VG(VALGRIND_DESTROY_MEMPOOL(pool));
}

struct anv_state
anv_state_pool_alloc(struct anv_state_pool *pool, size_t size, size_t align)
{
   unsigned size_log2 = ilog2_round_up(size < align ? align : size);
   assert(size_log2 <= ANV_MAX_STATE_SIZE_LOG2);
   if (size_log2 < ANV_MIN_STATE_SIZE_LOG2)
      size_log2 = ANV_MIN_STATE_SIZE_LOG2;
   unsigned bucket = size_log2 - ANV_MIN_STATE_SIZE_LOG2;

   struct anv_state state;
   state.alloc_size = 1 << size_log2;
   state.offset = anv_fixed_size_state_pool_alloc(&pool->buckets[bucket],
                                                  pool->block_pool);
   state.map = pool->block_pool->map + state.offset;
   VG(VALGRIND_MEMPOOL_ALLOC(pool, state.map, size));
   return state;
}

void
anv_state_pool_free(struct anv_state_pool *pool, struct anv_state state)
{
   assert(util_is_power_of_two(state.alloc_size));
   unsigned size_log2 = ilog2_round_up(state.alloc_size);
   assert(size_log2 >= ANV_MIN_STATE_SIZE_LOG2 &&
          size_log2 <= ANV_MAX_STATE_SIZE_LOG2);
   unsigned bucket = size_log2 - ANV_MIN_STATE_SIZE_LOG2;

   VG(VALGRIND_MEMPOOL_FREE(pool, state.map));
   anv_fixed_size_state_pool_free(&pool->buckets[bucket],
                                  pool->block_pool, state.offset);
}

#define NULL_BLOCK 1
struct anv_state_stream_block {
   /* The next block */
   struct anv_state_stream_block *next;

   /* The offset into the block pool at which this block starts */
   uint32_t offset;

#ifdef HAVE_VALGRIND
   /* A pointer to the first user-allocated thing in this block.  This is
    * what valgrind sees as the start of the block.
    */
   void *_vg_ptr;
#endif
};

/* The state stream allocator is a one-shot, single threaded allocator for
 * variable sized blocks.  We use it for allocating dynamic state.
 */
void
anv_state_stream_init(struct anv_state_stream *stream,
                      struct anv_block_pool *block_pool)
{
   stream->block_pool = block_pool;
   stream->block = NULL;

   /* Ensure that next + whatever > end.  This way the first call to
    * state_stream_alloc fetches a new block.
    */
   stream->next = 1;
   stream->end = 0;

   VG(VALGRIND_CREATE_MEMPOOL(stream, 0, false));
}

void
anv_state_stream_finish(struct anv_state_stream *stream)
{
   VG(const uint32_t block_size = stream->block_pool->block_size);

   struct anv_state_stream_block *next = stream->block;
   while (next != NULL) {
      struct anv_state_stream_block sb = VG_NOACCESS_READ(next);
      VG(VALGRIND_MEMPOOL_FREE(stream, sb._vg_ptr));
      VG(VALGRIND_MAKE_MEM_UNDEFINED(next, block_size));
      anv_block_pool_free(stream->block_pool, sb.offset);
      next = sb.next;
   }

   VG(VALGRIND_DESTROY_MEMPOOL(stream));
}

struct anv_state
anv_state_stream_alloc(struct anv_state_stream *stream,
                       uint32_t size, uint32_t alignment)
{
   struct anv_state_stream_block *sb = stream->block;

   struct anv_state state;

   state.offset = align_u32(stream->next, alignment);
   if (state.offset + size > stream->end) {
      uint32_t block = anv_block_pool_alloc(stream->block_pool);
      sb = stream->block_pool->map + block;

      VG(VALGRIND_MAKE_MEM_UNDEFINED(sb, sizeof(*sb)));
      sb->next = stream->block;
      sb->offset = block;
      VG(sb->_vg_ptr = NULL);
      VG(VALGRIND_MAKE_MEM_NOACCESS(sb, stream->block_pool->block_size));

      stream->block = sb;
      stream->start = block;
      stream->next = block + sizeof(*sb);
      stream->end = block + stream->block_pool->block_size;

      state.offset = align_u32(stream->next, alignment);
      assert(state.offset + size <= stream->end);
   }

   assert(state.offset > stream->start);
   state.map = (void *)sb + (state.offset - stream->start);
   state.alloc_size = size;

#ifdef HAVE_VALGRIND
   void *vg_ptr = VG_NOACCESS_READ(&sb->_vg_ptr);
   if (vg_ptr == NULL) {
      vg_ptr = state.map;
      VG_NOACCESS_WRITE(&sb->_vg_ptr, vg_ptr);
      VALGRIND_MEMPOOL_ALLOC(stream, vg_ptr, size);
   } else {
      void *state_end = state.map + state.alloc_size;
      /* This only updates the mempool.  The newly allocated chunk is still
       * marked as NOACCESS. */
      VALGRIND_MEMPOOL_CHANGE(stream, vg_ptr, vg_ptr, state_end - vg_ptr);
      /* Mark the newly allocated chunk as undefined */
      VALGRIND_MAKE_MEM_UNDEFINED(state.map, state.alloc_size);
   }
#endif

   stream->next = state.offset + size;

   return state;
}

struct bo_pool_bo_link {
   struct bo_pool_bo_link *next;
   struct anv_bo bo;
};

void
anv_bo_pool_init(struct anv_bo_pool *pool, struct anv_device *device)
{
   pool->device = device;
   memset(pool->free_list, 0, sizeof(pool->free_list));

   VG(VALGRIND_CREATE_MEMPOOL(pool, 0, false));
}

void
anv_bo_pool_finish(struct anv_bo_pool *pool)
{
   for (unsigned i = 0; i < ARRAY_SIZE(pool->free_list); i++) {
      struct bo_pool_bo_link *link = PFL_PTR(pool->free_list[i]);
      while (link != NULL) {
         struct bo_pool_bo_link link_copy = VG_NOACCESS_READ(link);

         anv_gem_munmap(link_copy.bo.map, link_copy.bo.size);
         anv_gem_close(pool->device, link_copy.bo.gem_handle);
         link = link_copy.next;
      }
   }

   VG(VALGRIND_DESTROY_MEMPOOL(pool));
}

VkResult
anv_bo_pool_alloc(struct anv_bo_pool *pool, struct anv_bo *bo, uint32_t size)
{
   VkResult result;

   const unsigned size_log2 = size < 4096 ? 12 : ilog2_round_up(size);
   const unsigned pow2_size = 1 << size_log2;
   const unsigned bucket = size_log2 - 12;
   assert(bucket < ARRAY_SIZE(pool->free_list));

   void *next_free_void;
   if (anv_ptr_free_list_pop(&pool->free_list[bucket], &next_free_void)) {
      struct bo_pool_bo_link *next_free = next_free_void;
      *bo = VG_NOACCESS_READ(&next_free->bo);
      assert(bo->gem_handle);
      assert(bo->map == next_free);
      assert(size <= bo->size);

      VG(VALGRIND_MEMPOOL_ALLOC(pool, bo->map, size));

      return VK_SUCCESS;
   }

   struct anv_bo new_bo;

   result = anv_bo_init_new(&new_bo, pool->device, pow2_size);
   if (result != VK_SUCCESS)
      return result;

   assert(new_bo.size == pow2_size);

   new_bo.map = anv_gem_mmap(pool->device, new_bo.gem_handle, 0, pow2_size, 0);
   if (new_bo.map == NULL) {
      anv_gem_close(pool->device, new_bo.gem_handle);
      return vk_error(VK_ERROR_MEMORY_MAP_FAILED);
   }

   *bo = new_bo;

   VG(VALGRIND_MEMPOOL_ALLOC(pool, bo->map, size));

   return VK_SUCCESS;
}

void
anv_bo_pool_free(struct anv_bo_pool *pool, const struct anv_bo *bo_in)
{
   /* Make a copy in case the anv_bo happens to be storred in the BO */
   struct anv_bo bo = *bo_in;

   VG(VALGRIND_MEMPOOL_FREE(pool, bo.map));

   struct bo_pool_bo_link *link = bo.map;
   VG_NOACCESS_WRITE(&link->bo, bo);

   assert(util_is_power_of_two(bo.size));
   const unsigned size_log2 = ilog2_round_up(bo.size);
   const unsigned bucket = size_log2 - 12;
   assert(bucket < ARRAY_SIZE(pool->free_list));

   anv_ptr_free_list_push(&pool->free_list[bucket], link);
}

// Scratch pool

void
anv_scratch_pool_init(struct anv_device *device, struct anv_scratch_pool *pool)
{
   memset(pool, 0, sizeof(*pool));
}

void
anv_scratch_pool_finish(struct anv_device *device, struct anv_scratch_pool *pool)
{
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      for (unsigned i = 0; i < 16; i++) {
         struct anv_scratch_bo *bo = &pool->bos[i][s];
         if (bo->exists > 0)
            anv_gem_close(device, bo->bo.gem_handle);
      }
   }
}

struct anv_bo *
anv_scratch_pool_alloc(struct anv_device *device, struct anv_scratch_pool *pool,
                       gl_shader_stage stage, unsigned per_thread_scratch)
{
   if (per_thread_scratch == 0)
      return NULL;

   unsigned scratch_size_log2 = ffs(per_thread_scratch / 2048);
   assert(scratch_size_log2 < 16);

   struct anv_scratch_bo *bo = &pool->bos[scratch_size_log2][stage];

   /* We can use "exists" to shortcut and ignore the critical section */
   if (bo->exists)
      return &bo->bo;

   pthread_mutex_lock(&device->mutex);

   __sync_synchronize();
   if (bo->exists)
      return &bo->bo;

   const struct anv_physical_device *physical_device =
      &device->instance->physicalDevice;
   const struct gen_device_info *devinfo = &physical_device->info;

   /* WaCSScratchSize:hsw
    *
    * Haswell's scratch space address calculation appears to be sparse
    * rather than tightly packed. The Thread ID has bits indicating which
    * subslice, EU within a subslice, and thread within an EU it is.
    * There's a maximum of two slices and two subslices, so these can be
    * stored with a single bit. Even though there are only 10 EUs per
    * subslice, this is stored in 4 bits, so there's an effective maximum
    * value of 16 EUs. Similarly, although there are only 7 threads per EU,
    * this is stored in a 3 bit number, giving an effective maximum value
    * of 8 threads per EU.
    *
    * This means that we need to use 16 * 8 instead of 10 * 7 for the
    * number of threads per subslice.
    */
   const unsigned subslices = MAX2(physical_device->subslice_total, 1);
   const unsigned scratch_ids_per_subslice =
      device->info.is_haswell ? 16 * 8 : devinfo->max_cs_threads;

   uint32_t max_threads[] = {
      [MESA_SHADER_VERTEX]           = devinfo->max_vs_threads,
      [MESA_SHADER_TESS_CTRL]        = devinfo->max_tcs_threads,
      [MESA_SHADER_TESS_EVAL]        = devinfo->max_tes_threads,
      [MESA_SHADER_GEOMETRY]         = devinfo->max_gs_threads,
      [MESA_SHADER_FRAGMENT]         = devinfo->max_wm_threads,
      [MESA_SHADER_COMPUTE]          = scratch_ids_per_subslice * subslices,
   };

   uint32_t size = per_thread_scratch * max_threads[stage];

   anv_bo_init_new(&bo->bo, device, size);

   /* Even though the Scratch base pointers in 3DSTATE_*S are 64 bits, they
    * are still relative to the general state base address.  When we emit
    * STATE_BASE_ADDRESS, we set general state base address to 0 and the size
    * to the maximum (1 page under 4GB).  This allows us to just place the
    * scratch buffers anywhere we wish in the bottom 32 bits of address space
    * and just set the scratch base pointer in 3DSTATE_*S using a relocation.
    * However, in order to do so, we need to ensure that the kernel does not
    * place the scratch BO above the 32-bit boundary.
    *
    * NOTE: Technically, it can't go "anywhere" because the top page is off
    * limits.  However, when EXEC_OBJECT_SUPPORTS_48B_ADDRESS is set, the
    * kernel allocates space using
    *
    *    end = min_t(u64, end, (1ULL << 32) - I915_GTT_PAGE_SIZE);
    *
    * so nothing will ever touch the top page.
    */
   bo->bo.flags &= ~EXEC_OBJECT_SUPPORTS_48B_ADDRESS;

   /* Set the exists last because it may be read by other threads */
   __sync_synchronize();
   bo->exists = true;

   pthread_mutex_unlock(&device->mutex);

   return &bo->bo;
}

struct anv_cached_bo {
   struct anv_bo bo;

   uint32_t refcount;
};

VkResult
anv_bo_cache_init(struct anv_bo_cache *cache)
{
   cache->bo_map = _mesa_hash_table_create(NULL, _mesa_hash_pointer,
                                           _mesa_key_pointer_equal);
   if (!cache->bo_map)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   if (pthread_mutex_init(&cache->mutex, NULL)) {
      _mesa_hash_table_destroy(cache->bo_map, NULL);
      return vk_errorf(VK_ERROR_OUT_OF_HOST_MEMORY,
                       "pthread_mutex_init failed: %m");
   }

   return VK_SUCCESS;
}

void
anv_bo_cache_finish(struct anv_bo_cache *cache)
{
   _mesa_hash_table_destroy(cache->bo_map, NULL);
   pthread_mutex_destroy(&cache->mutex);
}

static struct anv_cached_bo *
anv_bo_cache_lookup_locked(struct anv_bo_cache *cache, uint32_t gem_handle)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(cache->bo_map,
                              (const void *)(uintptr_t)gem_handle);
   if (!entry)
      return NULL;

   struct anv_cached_bo *bo = (struct anv_cached_bo *)entry->data;
   assert(bo->bo.gem_handle == gem_handle);

   return bo;
}

static struct anv_bo *
anv_bo_cache_lookup(struct anv_bo_cache *cache, uint32_t gem_handle)
{
   pthread_mutex_lock(&cache->mutex);

   struct anv_cached_bo *bo = anv_bo_cache_lookup_locked(cache, gem_handle);

   pthread_mutex_unlock(&cache->mutex);

   return bo ? &bo->bo : NULL;
}

VkResult
anv_bo_cache_alloc(struct anv_device *device,
                   struct anv_bo_cache *cache,
                   uint64_t size, struct anv_bo **bo_out)
{
   struct anv_cached_bo *bo =
      vk_alloc(&device->alloc, sizeof(struct anv_cached_bo), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!bo)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   bo->refcount = 1;

   /* The kernel is going to give us whole pages anyway */
   size = align_u64(size, 4096);

   VkResult result = anv_bo_init_new(&bo->bo, device, size);
   if (result != VK_SUCCESS) {
      vk_free(&device->alloc, bo);
      return result;
   }

   assert(bo->bo.gem_handle);

   pthread_mutex_lock(&cache->mutex);

   _mesa_hash_table_insert(cache->bo_map,
                           (void *)(uintptr_t)bo->bo.gem_handle, bo);

   pthread_mutex_unlock(&cache->mutex);

   *bo_out = &bo->bo;

   return VK_SUCCESS;
}

VkResult
anv_bo_cache_import(struct anv_device *device,
                    struct anv_bo_cache *cache,
                    int fd, uint64_t size, struct anv_bo **bo_out)
{
   pthread_mutex_lock(&cache->mutex);

   /* The kernel is going to give us whole pages anyway */
   size = align_u64(size, 4096);

   uint32_t gem_handle = anv_gem_fd_to_handle(device, fd);
   if (!gem_handle) {
      pthread_mutex_unlock(&cache->mutex);
      return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHX);
   }

   struct anv_cached_bo *bo = anv_bo_cache_lookup_locked(cache, gem_handle);
   if (bo) {
      if (bo->bo.size != size) {
         pthread_mutex_unlock(&cache->mutex);
         return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHX);
      }
      __sync_fetch_and_add(&bo->refcount, 1);
   } else {
      /* For security purposes, we reject BO imports where the size does not
       * match exactly.  This prevents a malicious client from passing a
       * buffer to a trusted client, lying about the size, and telling the
       * trusted client to try and texture from an image that goes
       * out-of-bounds.  This sort of thing could lead to GPU hangs or worse
       * in the trusted client.  The trusted client can protect itself against
       * this sort of attack but only if it can trust the buffer size.
       */
      off_t import_size = lseek(fd, 0, SEEK_END);
      if (import_size == (off_t)-1 /* || import_size != size */) {
         anv_gem_close(device, gem_handle);
         pthread_mutex_unlock(&cache->mutex);
         return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHX);
      }

      bo = vk_alloc(&device->alloc, sizeof(struct anv_cached_bo), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!bo) {
         anv_gem_close(device, gem_handle);
         pthread_mutex_unlock(&cache->mutex);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      bo->refcount = 1;

      anv_bo_init(&bo->bo, gem_handle, size);

      if (device->instance->physicalDevice.supports_48bit_addresses)
         bo->bo.flags |= EXEC_OBJECT_SUPPORTS_48B_ADDRESS;

      if (device->instance->physicalDevice.has_exec_async)
         bo->bo.flags |= EXEC_OBJECT_ASYNC;

      _mesa_hash_table_insert(cache->bo_map, (void *)(uintptr_t)gem_handle, bo);
   }

   pthread_mutex_unlock(&cache->mutex);

   /* From the Vulkan spec:
    *
    *    "Importing memory from a file descriptor transfers ownership of
    *    the file descriptor from the application to the Vulkan
    *    implementation. The application must not perform any operations on
    *    the file descriptor after a successful import."
    *
    * If the import fails, we leave the file descriptor open.
    */
   close(fd);

   *bo_out = &bo->bo;

   return VK_SUCCESS;
}

VkResult
anv_bo_cache_export(struct anv_device *device,
                    struct anv_bo_cache *cache,
                    struct anv_bo *bo_in, int *fd_out)
{
   assert(anv_bo_cache_lookup(cache, bo_in->gem_handle) == bo_in);
   struct anv_cached_bo *bo = (struct anv_cached_bo *)bo_in;

   int fd = anv_gem_handle_to_fd(device, bo->bo.gem_handle);
   if (fd < 0)
      return vk_error(VK_ERROR_TOO_MANY_OBJECTS);

   *fd_out = fd;

   return VK_SUCCESS;
}

static bool
atomic_dec_not_one(uint32_t *counter)
{
   uint32_t old, val;

   val = *counter;
   while (1) {
      if (val == 1)
         return false;

      old = __sync_val_compare_and_swap(counter, val, val - 1);
      if (old == val)
         return true;

      val = old;
   }
}

void
anv_bo_cache_release(struct anv_device *device,
                     struct anv_bo_cache *cache,
                     struct anv_bo *bo_in)
{
   assert(anv_bo_cache_lookup(cache, bo_in->gem_handle) == bo_in);
   struct anv_cached_bo *bo = (struct anv_cached_bo *)bo_in;

   /* Try to decrement the counter but don't go below one.  If this succeeds
    * then the refcount has been decremented and we are not the last
    * reference.
    */
   if (atomic_dec_not_one(&bo->refcount))
      return;

   pthread_mutex_lock(&cache->mutex);

   /* We are probably the last reference since our attempt to decrement above
    * failed.  However, we can't actually know until we are inside the mutex.
    * Otherwise, someone could import the BO between the decrement and our
    * taking the mutex.
    */
   if (unlikely(__sync_sub_and_fetch(&bo->refcount, 1) > 0)) {
      /* Turns out we're not the last reference.  Unlock and bail. */
      pthread_mutex_unlock(&cache->mutex);
      return;
   }

   struct hash_entry *entry =
      _mesa_hash_table_search(cache->bo_map,
                              (const void *)(uintptr_t)bo->bo.gem_handle);
   assert(entry);
   _mesa_hash_table_remove(cache->bo_map, entry);

   if (bo->bo.map)
      anv_gem_munmap(bo->bo.map, bo->bo.size);

   anv_gem_close(device, bo->bo.gem_handle);

   /* Don't unlock until we've actually closed the BO.  The whole point of
    * the BO cache is to ensure that we correctly handle races with creating
    * and releasing GEM handles and we don't want to let someone import the BO
    * again between mutex unlock and closing the GEM handle.
    */
   pthread_mutex_unlock(&cache->mutex);

   vk_free(&device->alloc, bo);
}
