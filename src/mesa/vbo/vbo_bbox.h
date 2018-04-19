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

#ifndef _VBO_BBOX_H_
#define _VBO_BBOX_H_

#include <stdio.h>
#include "main/arrayobj.h"
#include "main/glheader.h"
#include "main/context.h"
#include "main/state.h"
#include "main/varray.h"
#include "main/bufferobj.h"
#include "main/arrayobj.h"
#include "main/enums.h"
#include "main/macros.h"
#include "main/transformfeedback.h"
#include "main/mtypes.h"
#include "compiler/glsl/ir_uniform.h"
#include "main/shaderapi.h"
#include "main/uniforms.h"
#include "sys/param.h"
#include "program/prog_cache.h"
/* For Intrinsic functions */
#include <smmintrin.h>
#include <tmmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>

#if defined(__ANDROID__) || defined(ANDROID)
#include <cutils/log.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FRUSTUM_PLANE_COUNT  6
#define BBOX_CALC_MIN_DELAY  2
#define BBOX_CALC_MAX_DELAY  5

#undef LOG_TAG
#define LOG_TAG "MESA_BBOX_LOG"

/* Default, driver will select mode for given GPU and OS */
#define MESA_BBOX_ENABLE_AUTO               -1
/* Disable BBOX */
#define MESA_BBOX_ENABLE_OFF                 0
/* Enable BBOX, bot clipping */
#define MESA_BBOX_ENABLE_SMART               1
/* Enable BBOX, clipping will be done regardless of GPU utilization */
#define MESA_BBOX_ENABLE_FORCE_CLIPPING      2
/* Enable BBOX, force immediate bbox recalculation and clipping */
#define MESA_BBOX_ENABLE_FORCE_RECALC        3

/**
 * MESA BBOX PRINTS
 * Uncomment below line to enable debugging logs
 */
#define MESA_BBOX_DEBUG 0

#if defined(__ANDROID__) || defined (ANDROID)

#if MESA_BBOX_DEBUG == 2
#define MESA_BBOX_PRINT(...) ALOGE(__VA_ARGS__)
#define MESA_BBOX(...) ALOGE(__VA_ARGS__)

#elif MESA_BBOX_DEBUG == 1
#define MESA_BBOX_PRINT(...) ALOGE(__VA_ARGS__)
#define MESA_BBOX(...)

#else
#define MESA_BBOX_PRINT(...)
#define MESA_BBOX(...)
#endif //MESA_BBOX_DEBUG

#else //ANDROID

#if MESA_BBOX_DEBUG == 2
#define MESA_BBOX_PRINT(...) printf(__VA_ARGS__)
#define MESA_BBOX(...) printf(__VA_ARGS__)

#elif MESA_BBOX_DEBUG == 1
#define MESA_BBOX_PRINT(...) printf(__VA_ARGS__)
#define MESA_BBOX(...)

#else
#define MESA_BBOX_PRINT(...)
#define MESA_BBOX(...)
#endif //MESA_BBOX_DEBUG

#endif //ANDROID

/**
 * MESA Bbox options environment variables
 */
int env_opt_val;
const char *env_opt;

typedef struct vbo_bbox_env_variable {
   GLuint bbox_min_vrtx_count;
   GLuint bbox_enable;
   GLuint bbox_split_size;
   GLuint bbox_trace_level;
} bbox_env;
bbox_env mesa_bbox_env_variables;

void
vbo_bbox_init(struct gl_context *const gc);

void
vbo_bbox_free(struct gl_context *const gc);


void
vbo_bbox_element_buffer_update(struct gl_context *const gc,
                               struct gl_buffer_object *buffer,
                               const void* data,
                               int offset,
                               int size);

void
vbo_validated_drawrangeelements(struct gl_context *ctx,
                                GLenum mode,
                                GLboolean index_bounds_valid,
                                GLuint start,
                                GLuint end,
                                GLsizei count,
                                GLenum type,
                                const GLvoid * indices,
                                GLint basevertex,
                                GLuint numInstances,
                                GLuint baseInstance);

void
vbo_bbox_drawelements(struct gl_context *ctx,
                      GLenum mode,
                      GLboolean index_bounds_valid,
                      GLuint start,
                      GLuint end,
                      GLsizei count,
                      GLenum type,
                      const GLvoid * indices,
                      GLint basevertex,
                      GLuint numInstances,
                      GLuint baseInstance);

/**
 * Segment Functions
 */
typedef struct gl_segment
{
  GLint Left;
  GLint Right;

}segment;

/**
 * Clip algorithm result
 */
enum vbo_bbox_clip_result
{
    BBOX_CLIP_OUTSIDE=0,
    BBOX_CLIP_INTERSECT=1,
    BBOX_CLIP_INSIDE=2,
    BBOX_CLIP_DEGEN = 3,
    BBOX_CLIP_ERROR = 4,
};

typedef struct gl_matrixRec
{
    GLfloat melem[4][4];
    GLenum    matrixType;
} gl_matrix;

/**
 * Structure to describe plane
 */
typedef struct vbo_bbox_frustum_plane
{
    GLfloat a, b, c, d;
} vbo_bbox_frustum_plane;


/**
 * Planes and octants with their normals
 */
typedef struct vbo_bbox_frustum
{
    const unsigned char planeCount;
    struct vbo_bbox_frustum_plane plane[FRUSTUM_PLANE_COUNT];
    unsigned char octant[FRUSTUM_PLANE_COUNT];
} vbo_bbox_frustum;

/*
 * Axis Aligned Bounding Box
 */
typedef union vbo_vec4f {
    GLfloat data[4];
    struct {
        GLfloat x, y, z, w;
    };
} vbo_vec4f;

/*
 * Oriented Bounding Box
 */
typedef struct oriented_bounding_box {
        GLfloat x,y,z,w;
} oriented_box;

/*
 * Spherical Bounding volume
 */
typedef struct spherical_bounding_volume {
        GLfloat x,y,z,r; // x^2+y^2+z^2 = r^2
} spherical_volume;

/*
 * 8-Discrete oriented polytopes
 */
typedef struct dop_bounding_volume {
        GLfloat x,y,z,r,a,b;//TBD Not sure of the representation for 8-DOP yet!
} dop_volume;

/*
 * Bounding volumes for AABB, OBB, SPHERE, DOP etc
 */
typedef union bounding_volume_info {
    vbo_vec4f vert_vec4[8]; /* Bbox mix man coordinates */
    oriented_box obb[8];
    spherical_volume sphere;
    dop_volume dop[8];
} bounding_volume_info;


typedef struct vbo_bbox_cache_key
{
    /* From API call */
    GLenum mode;         /* GL_TRAINGLES are only mode supported currently */
    GLsizei count;       /* Number if indices in draw call */
    GLenum indices_type;  /* must be GL_UNSIGNED_SHORT for now */
    GLuint type_size;
    GLuint type_sizeShift;
    GLint   indices;     /* Offset to index VBO */
    GLint basevertex;    /* Only 0 supported for now. */

    /* VBO objects names */
    GLuint element_buf_name;
    GLuint vertex_buf_name;

    /* Vertex position attribute configuration */
    GLint    offset;
    GLint    size;          /* Size of attribute, must be 3 for now */
    GLenum   vertDataType;  /* Must be GL_FLOAT */
    GLsizei  stride;        /* Any */
    GLuint   elementStride;
} vbo_bbox_cache_key;

typedef struct bounding_info
{
    int vert_count;   /* Number of vertices this bbox covers */
    int start_offset;   /* Start offset for this bbox */
    bool is_degenerate; /* Triangle can not be formed */
    enum vbo_bbox_clip_result clip_result;

    bounding_volume_info bounding_volume; /* Bbox mix man coordinates */

} bounding_info;

/**
 * Cached information about (multiple) bounding boxes
 */
struct vbo_bbox_cache_data
{
    /* Valid data indicator, will be set to false if VBO have been
     * modified by application
     */
    bool valid;

    /* Indicates if bounding boxes were calculated for this geometry */
    bool need_new_calculation;

    /* Controls delay with which bounding boxes are calculated */
    int init_delaycnt;
    int init_delaylimit;

    /* Data change indicator for VBOs */
    int vertpos_vbo_changecnt;
    int indices_vbo_changecnt;

    /* How many bounding boxes are stored for this geometry */
    int sub_bbox_cnt;

    /* How many draws call the bbox was effective */
    int drawcnt_bbox_helped;

    int last_use_frame;

    GLuint hash;

    unsigned keysize;

    vbo_bbox_cache_key *key;

    bool mvp_valid;
    GLmatrix mvpin;
    GLfloat mvp[16];
    vbo_bbox_frustum frustum;

    /* Pointer to array of bboxes */
    bounding_info* sub_bbox_array;

    /* Bounding box that covers whole geometry */
    bounding_info full_box;

    struct vbo_bbox_cache_data *next;
};

typedef struct mesa_bbox_cache {
     struct vbo_bbox_cache_data **items;
     struct vbo_bbox_cache_data *last;
     GLuint size, n_items;
} mesa_bbox_cache;

typedef struct mesa_bbox_opt
{
    mesa_bbox_cache * cache;
    GLuint calc_delay;
} mesa_bbox_opt;

/**
 * VBO BBox Cache functions, replica of program cache functions
 *
 */
struct mesa_bbox_cache *
_mesa_new_bbox_cache(void);

void
_mesa_delete_bbox_cache(struct gl_context *ctx,
                        struct mesa_bbox_cache *cache);

struct vbo_bbox_cache_data *
_mesa_search_bbox_cache(struct mesa_bbox_cache *cache,
                        const void *key, GLuint keysize);

void
_mesa_bbox_cache_insert(struct gl_context *ctx,
                        struct mesa_bbox_cache *cache,
                        const void *key,
                        GLuint keysize,
                        struct vbo_bbox_cache_data *CachedData);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //_VBO_BBOX_H_
