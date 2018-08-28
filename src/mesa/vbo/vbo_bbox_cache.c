
/*
 * Copyright © 2018 Intel Corporation
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

/**
 * \brief  VBO BBOX module implementation
 * \author Kedar Karanje
 */

#include "main/imports.h"
#include "main/mtypes.h"
#include "vbo/vbo_bbox.h"

#define CACHE_SIZE 17

/**
 * Compute hash index from state key.
 */
static GLuint
bbox_hash_key(const void *key, GLuint key_size)
{
   const GLuint *ikey = (const GLuint *) key;
   GLuint hash = 0, i;

   assert(key_size >= 4);

   /* Make a slightly better attempt at a hash function:
    */
   for (i = 0; i < key_size / sizeof(*ikey); i++)
   {
      hash += ikey[i];
      hash += (hash << 10);
      hash ^= (hash >> 6);
   }

   return hash;
}


/**
 * Rebuild/expand the hash table to accommodate more entries
 */
static void
bbox_rehash(struct mesa_bbox_cache *cache)
{
   struct vbo_bbox_cache_data **items;
   struct vbo_bbox_cache_data *c, *next;
   GLuint size, i;

   cache->last = NULL;

   size = cache->size * 3;
   items = calloc(size, sizeof(*items));

   for (i = 0; i < cache->size; i++)
      for (c = cache->items[i]; c; c = next) {
         next = c->next;
         c->next = items[c->hash % size];
         items[c->hash % size] = c;
      }

   free(cache->items);
   cache->items = items;
   cache->size = size;
}


static void
bbox_clear_cache(struct gl_context *ctx, mesa_bbox_cache *cache)
{
   struct vbo_bbox_cache_data *c, *next;
   GLuint i;

   cache->last = NULL;

   for (i = 0; i < cache->size; i++) {
      for (c = cache->items[i]; c; c = next) {
         next = c->next;
         free(c->key);
         free(c->sub_bbox_array);
         free(c);
      }
      cache->items[i] = NULL;
   }
   cache->n_items = 0;
}



mesa_bbox_cache *
_mesa_new_bbox_cache(void)
{
   mesa_bbox_cache *cache = CALLOC_STRUCT(mesa_bbox_cache);
   if (cache) {
      cache->size = CACHE_SIZE;
      cache->items = calloc(cache->size, sizeof(struct vbo_bbox_cache_data));
      if (!cache->items) {
         MESA_BBOX("Func:%s cache-size=%d "
                   "Cannot allocate items freeing cache\n",
                   __func__,cache->size);
         free(cache);
         return NULL;
      }
      MESA_BBOX("Func:%s cache:%#x cache->size=%d \n",
                 __func__,cache,cache->size);
      return cache;
   }
   else {
      MESA_BBOX("cache is Null in Func:%s\n",__func__);
      return cache;
   }
}


void
_mesa_delete_bbox_cache(struct gl_context *ctx, mesa_bbox_cache *cache)
{
   bbox_clear_cache(ctx, cache);
   free(cache->items);
   free(cache);
}

struct vbo_bbox_cache_data *
_mesa_search_bbox_cache(mesa_bbox_cache *cache,
                           const void *key, GLuint keysize)
{
   MESA_BBOX("Func:%s cache:%#x \n",__func__,cache);
   if (cache->last &&
       cache->last->key->mode == ((vbo_bbox_cache_key *)key)->mode &&
       cache->last->key->count == ((vbo_bbox_cache_key *)key)->count &&
       cache->last->key->indices == ((vbo_bbox_cache_key *)key)->indices) {
      return cache->last;
   }
   else {
      const GLuint hash = bbox_hash_key(key, keysize);
      struct vbo_bbox_cache_data *c;
      MESA_BBOX("cache:%#x,hash:%d,cache->size:%d\n",cache,hash,cache->size);
      for (c = cache->items[hash % cache->size]; c; c = c->next) {
         if (c->hash == hash &&
            c->key->mode == ((vbo_bbox_cache_key *)key)->mode &&
            c->key->count == ((vbo_bbox_cache_key *)key)->count &&
            c->key->indices == ((vbo_bbox_cache_key *)key)->indices) {
            cache->last = c;
            return c;
         }
      }
      return NULL;
   }
}


void
_mesa_bbox_cache_insert(struct gl_context *ctx,struct mesa_bbox_cache *cache,
                        const void *key, GLuint keysize,
                        struct vbo_bbox_cache_data *CachedData)
{
   const GLuint hash = bbox_hash_key(key, keysize);

   CachedData->hash = hash;

   CachedData->key = calloc(1, keysize);
   memcpy(CachedData->key, key, keysize);
   CachedData->keysize = keysize;

   if (cache->n_items > cache->size * 1.5) {
      if (cache->size < 1000)
         bbox_rehash(cache);
      else
         bbox_clear_cache(ctx, cache);
   }

   cache->n_items++;
   CachedData->next = cache->items[hash % cache->size];
   cache->items[hash % cache->size] = CachedData;
}
