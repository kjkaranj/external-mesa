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

#pragma GCC optimize (0)
#include "vbo_bbox.h"
#include <fcntl.h>
#define  BBOX_MAX_FRAMES_TO_DYNAMICALLY_UPDATE (5)
#define  BBOX_MIN_EFFECTIVE_DRAWS_TO_DYNAMICALLY_UPDATE (5)
#define  BBOX_MIN_VAL_FOR_EFFECTIVE_DRAWS_COUNTER (-100)
#define  BBOX_MAX_VAL_FOR_EFFECTIVE_DRAWS_COUNTER (100)
#define  BBOX_MIN_GPU_HEAD_ROOM_TO_PROCESS_ELEMENT_BUFFER (100)

/**
  *  Min part of split draw that we want to drop
  */
#define  BBOX_MIN_SPLITTED_DRAW_TO_DROP (1)

#ifdef __AVX__
static __m256 fullMin;
static __m256 fullMax;
#else
static __m128 fullMin;
static __m128 fullMax;
#endif

/* Segment functions */

static inline GLboolean
intersect(const struct gl_segment *s_src,const struct gl_segment *s_tar)
{
   return (s_src->Left < s_tar->Right && s_src->Right > s_tar->Left);
}


static inline GLboolean
subsegment(const struct gl_segment *s_src,const struct gl_segment *s_tar)
{
  return (s_src->Left <= s_tar->Left && s_src->Right >= s_tar->Right);
}


static inline GLboolean
superset(const struct gl_segment *s_src,const struct gl_segment *s_tar)
{
  return subsegment(s_tar,s_src);
}


static void
normalize(vbo_bbox_frustum_plane *fr_plane)
{
        GLfloat a,b,c;
        a = fr_plane->a;
        b = fr_plane->b;
        c = fr_plane->c;

        GLfloat norm = 1.0f/sqrt(a*a + b*b + c*c);

        fr_plane->a *= norm;
        fr_plane->b *= norm;
        fr_plane->c *= norm;
        fr_plane->d *= norm;
};

static inline
int vbo_bbox_get_delay(struct mesa_bbox_opt *opt)
{
   if (mesa_bbox_env_variables.bbox_enable < MESA_BBOX_ENABLE_FORCE_RECALC) {
      if (opt->calc_delay > BBOX_CALC_MAX_DELAY) {
         opt->calc_delay = BBOX_CALC_MIN_DELAY;
      }
      return opt->calc_delay++;
   }
   else {
      return 0;
   }
}

static inline
bool vbo_init_sub_bbox_array(int bbox_count, struct vbo_bbox_cache_data* data)
{
   bool allocate_bbox = true;
   if(data->sub_bbox_array != NULL) {
      if(bbox_count == data->sub_bbox_cnt) {
         allocate_bbox = false;
      }
      else {
         free(data->sub_bbox_array);
         data->sub_bbox_array = NULL;
      }
   }
   if (allocate_bbox)
      data->sub_bbox_array = (struct bounding_info*) malloc(
                                    bbox_count * sizeof (struct bounding_info));

   if (!data->sub_bbox_array)
   return false;

   data->sub_bbox_cnt = bbox_count;
   return true;
}

static inline GLint
vbo_bbox_get_mvp(struct gl_context *ctx)
{
   struct gl_linked_shader * linked_shader =
                  ctx->Shader.ActiveProgram->_LinkedShaders[MESA_SHADER_VERTEX];

   return _mesa_GetUniformLocation(ctx->Shader.ActiveProgram->Name,
                                   linked_shader->linkedshaderMVP);
}

/*
 * Gets the currently linked shaders flag for simple shader
 *
 */
static inline int
vbo_is_simple_shader(struct gl_context *ctx)
{
   struct gl_linked_shader * linked_shader =
                  ctx->Shader.ActiveProgram->_LinkedShaders[MESA_SHADER_VERTEX];

   return linked_shader->linked_bbox_simple_shader;
}

/**
 * Get current VAO
 */
static inline struct gl_vertex_array_object*
vbo_get_current_vao(struct gl_context *const gc)
{
   assert(gc);
   struct gl_vertex_array_object* vao = gc->Array.VAO;
   assert(vao);

   return vao;
}

/**
 *  Returns the location of the "position" in the attributes of the
 *  currently active program
 */
static inline
int vbo_get_simple_vs_position_attr_location(
    struct gl_context *const gc)
{
   assert(gc);
   struct gl_linked_shader * linked_shader =
                   gc->Shader.ActiveProgram->_LinkedShaders[MESA_SHADER_VERTEX];
   GLint vertexPosLocation = _mesa_GetAttribLocation(
                           gc->Shader.ActiveProgram->Name,
                           linked_shader->linkedshaderVertPosition);
   if (vertexPosLocation >= 0)
      return vertexPosLocation;
   else
      return -1;
}

/**
 *   Get element-buffer handle of the current VAO
 */
static inline struct gl_buffer_object*
vbo_get_current_element_buffer(struct gl_context *const gc)
{
   assert(gc);
   struct gl_buffer_object* element_buffer =
                                        vbo_get_current_vao(gc)->IndexBufferObj;
   assert(element_buffer);
   return element_buffer;
}

/**
 *   Get vertex-binding of position from the current VAO
 */
static inline struct gl_vertex_buffer_binding*
vbo_get_current_vertex_buffer_binding_of_position(struct gl_context *const gc)
{
   assert(gc);
   struct gl_vertex_array_object* vao = vbo_get_current_vao(gc);

   GLbitfield mask = vao->_Enabled & vao->VertexAttribBufferMask;
   const struct gl_array_attributes *attrib_array =
                                           &vao->VertexAttrib[ffs(mask) - 1];
   struct gl_vertex_buffer_binding *buffer_binding =
                       &vao->BufferBinding[attrib_array->BufferBindingIndex];

   return buffer_binding;
}

/**
 * Get vertex-buffer handle of the current VAO
 */
static inline struct gl_buffer_object*
vbo_get_current_vertex_buffer(struct gl_context *const gc)
{
   assert(gc);
   struct gl_buffer_object* pVertexBuffer =
               vbo_get_current_vertex_buffer_binding_of_position(gc)->BufferObj;
   assert(pVertexBuffer);
   return pVertexBuffer;
}


/**
 * Condition to enter bounding box optimization
 */
static inline bool
vbo_bbox_check_supported_draw_call(struct gl_context *const gc,
                                   GLenum mode, GLsizei count, GLenum type,
                                   const GLvoid *indices, GLint basevertex)
{

   assert(gc);
   int shader_scenario;
   struct gl_linked_shader *_LinkedShaders;

   /* Check if the minimum vertex count is met. */
   if (count < (GLsizei) mesa_bbox_env_variables.bbox_min_vrtx_count) {
     /* Count is most common cause to bail out form optimization
      * so should be first.
      */
     MESA_BBOX("Aborting MESA_BBOX :%d: Vertex count too small, minimum count = %d\n",
                     count,mesa_bbox_env_variables.bbox_min_vrtx_count);
     return false;
   }

   if (mode != GL_TRIANGLES) {
      MESA_BBOX("Aborting MESA_BBOX :%d: Primitive mode is not GL_TRIANGLES, \
                      mode = %d\n", count, mode);
      return false;
   }

   /* Examine current shader */
   if (!gc->_Shader->ActiveProgram) {
      MESA_BBOX("Aborting MESA_BBOX:%d: No active GLSL program.\n", count);
      return false;
   }

   /* BASIC Shader scenario is when we have just VS & FS */
   if (gc->_Shader->CurrentProgram[MESA_SHADER_VERTEX] != NULL &&
       gc->_Shader->CurrentProgram[MESA_SHADER_FRAGMENT] != NULL &&
       gc->_Shader->CurrentProgram[MESA_SHADER_TESS_CTRL] == NULL &&
       gc->_Shader->CurrentProgram[MESA_SHADER_TESS_EVAL] == NULL &&
       gc->_Shader->CurrentProgram[MESA_SHADER_GEOMETRY] == NULL) {
      shader_scenario = 0;
   }
   else
     shader_scenario = 1;

   if (shader_scenario) {
     MESA_BBOX("Aborting MESA_BBOX:%d: GLSL program must contain only vertex and \
                     fragment shaders, shader scenario = \n", count );
     return false;
   }

   if (!vbo_is_simple_shader(gc)) {
      MESA_BBOX("Aborting MESA_BBOX:%d: GLSL vertex shader does not have simple \
                 position calculation \n", count);
      return false;
   }
   if (gc->Shader.ActiveProgram->_LinkedShaders[MESA_SHADER_VERTEX]) {
       _LinkedShaders =
               gc->Shader.ActiveProgram->_LinkedShaders[MESA_SHADER_VERTEX];

       MESA_BBOX("MVP:%s, VertPos:%s\n",_LinkedShaders->linkedshaderMVP,
                   _LinkedShaders->linkedshaderVertPosition);
   }

   /* Examine element buffer */
   struct gl_buffer_object* element_buffer = vbo_get_current_element_buffer(gc);
   if ((!element_buffer) || element_buffer->Name == 0) {
      MESA_BBOX("Aborting MESA_BBOX:%d: Element buffer name is 0\n", count);
      return false;
   }

   if (!(element_buffer->StorageFlags &
      (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))){
       MESA_BBOX("Aborting MESA_BBOX:%d: Element buffer not resident: %#x\n", count,
                     element_buffer->StorageFlags);
       return false;
   }

   /* Get VertexPosLocation */
   int vertexPosLocation = 0;
   if (gc->Shader.ActiveProgram)
      vertexPosLocation = vbo_get_simple_vs_position_attr_location(gc);
   if (vertexPosLocation < 0)
   {
      MESA_BBOX("Aborting MESA_BBOX:%d: VertexPosition Location is inValid:\n", count);
      return false;
   }

   struct gl_vertex_array_object*  vao = vbo_get_current_vao(gc);
   int posAttribMapMode =
            _mesa_vao_attribute_map[vao->_AttributeMapMode][vertexPosLocation];

   if (((vao->_Enabled >> posAttribMapMode) & 0x1) != 1)
   {
      MESA_BBOX("Aborting MESA_BBOX:%d: Vertex data does not come from VBO , GL-API:%d\n", count,gc->API);
//#if !defined(__ANDROID__) || !defined(ANDROID)
//This is not specific to Android but to the GLES API
      if (gc->API != API_OPENGLES && gc->API != API_OPENGLES2)
          return false;
//#endif
   }

   struct gl_buffer_object* vertexattrib_buffer  =
                            vbo_get_current_vertex_buffer(gc);
   if ((!vertexattrib_buffer) || vertexattrib_buffer->Name == 0) {
     MESA_BBOX("Aborting MESA_BBOX:%d: Vertex buffer %p name is %d\n", count,
                     vertexattrib_buffer,vertexattrib_buffer->Name);
#if !defined(__ANDROID__) || !defined(ANDROID)
     return false;
#endif
   }

   if (!(vertexattrib_buffer->StorageFlags &
        (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))){
       MESA_BBOX("Aborting MESA_BBOX:%d:Vertex buffer not resident %#x \n", count,
                     vertexattrib_buffer->StorageFlags);
       MESA_BBOX("Aborting MESA_BBOX:VAO AttributeMode is %d\n", vao->_AttributeMapMode);
#if !defined(__ANDROID__) || !defined(ANDROID)
       return false;
#endif
   }

    /* Examine vertex position attribute configuration */
   if (vao->VertexAttrib[posAttribMapMode].Enabled) {
      if (vao->VertexAttrib[posAttribMapMode].Size != 3)
      {
          MESA_BBOX("Aborting MESA_BBOX:%d: Vertex attrib size :%d, only 3 supported\n",
                         count, vao->VertexAttrib[VERT_ATTRIB_POS].Size);
#if !defined(__ANDROID__) || !defined(ANDROID)
          return false;
#endif
      }
      if (vao->VertexAttrib[posAttribMapMode].Type != GL_FLOAT)
      {
         MESA_BBOX("Aborting MESA_BBOX:%d: Vertex attrib type is %d, only GL_FLOAT \
                 supported\n", count, vao->VertexAttrib[VERT_ATTRIB_POS].Type);
         return false;
      }
   }

   if (type != GL_UNSIGNED_SHORT) {
      MESA_BBOX("Aborting MESA_BBOX:%d: type is %d, only GL_UNSIGNED_SHORT \
                 supported\n", count, type);
      return false;
   }
   if (basevertex != 0) {
      MESA_BBOX("Aborting MESA_BBOX:%d: basevertex is 0 \n", count);
      return false;
   }

   /* If size ==3 and type == GL_FLOAT, then element stride must be 12. */
   assert(vao->VertexAttrib[VERT_ATTRIB_POS].StrideB == 12);

   /* When transform feedback is capturing we cannot do early clipping since
   * xfb must write unclipped vertices
   * Note - we could check for IsCapturing() but that would require
   * more elaborate checking for VBO modifications.
   */
   if (gc->TransformFeedback.CurrentObject->Active) {
      MESA_BBOX("MESA_BBOX:%d: Transform feedback is active, \
                      cannot clip\n", count);
      return false;
   }
   return true;
}


/**
 *  Check condition to enter bounding box optimization and if draw call
 *  is suitable prepare key describing given geometry.
 */
static inline
void vbo_bbox_prepare_key(struct gl_context *const gc, GLenum mode,
                          GLsizei count, GLenum type, GLuint type_size,
                          const GLvoid *indices, GLint basevertex,
                          vbo_bbox_cache_key *key)
{
   assert(gc);

   /* Examine element buffer */
   struct gl_buffer_object* element_buffer = vbo_get_current_element_buffer(gc);
   struct gl_buffer_object* vertexattrib_buffer  =
                                           vbo_get_current_vertex_buffer(gc);
   struct gl_vertex_buffer_binding* vbinding =
                       vbo_get_current_vertex_buffer_binding_of_position(gc);

   memset(key,0,sizeof(vbo_bbox_cache_key));

   key->mode = mode;
   key->count = count;
   key->indices_type = type;
   key->type_size = type_size;
   key->indices = (GLint) (uintptr_t) indices;
   key->basevertex = basevertex;

   key->element_buf_name = element_buffer->Name;
   key->vertex_buf_name = vertexattrib_buffer->Name;

   key->offset = (GLint)vbinding->Offset;
   key->stride = vbinding->Stride;
}


/**
 *  Create a bounding box descriptor in a form of 8 correctly
 *  ordered vertex coordinates. The order of coordinates is significant.
 */
static
void vbo_bbox_create_bounding_box(float* const minVec3f, float* const maxVec3F,
                                  vbo_vec4f* vertices4)
{
   assert(minVec3f);
   assert(maxVec3F);
   assert(vertices4);

   float Xmin = minVec3f[0];
   float Ymin = minVec3f[1];
   float Zmin = minVec3f[2];
   float Wmin = 1.0f;

   float Xmax = maxVec3F[0];
   float Ymax = maxVec3F[1];
   float Zmax = maxVec3F[2];
   float Wmax = 1.0f;

   float* v = (float*)vertices4;
   int i = 0;
   v[i+0] = Xmin; v[i+1] = Ymin; v[i+2] = Zmin;v[i+3]=Wmin; i+=4;
   v[i+0] = Xmax; v[i+1] = Ymin; v[i+2] = Zmin;v[i+3]=Wmin; i+=4;
   v[i+0] = Xmin; v[i+1] = Ymax; v[i+2] = Zmin;v[i+3]=Wmin; i+=4;
   v[i+0] = Xmax; v[i+1] = Ymax; v[i+2] = Zmin;v[i+3]=Wmin; i+=4;

   v[i+0] = Xmin; v[i+1] = Ymin; v[i+2] = Zmax;v[i+3]=Wmax; i+=4;
   v[i+0] = Xmax; v[i+1] = Ymin; v[i+2] = Zmax;v[i+3]=Wmax; i+=4;
   v[i+0] = Xmin; v[i+1] = Ymax; v[i+2] = Zmax;v[i+3]=Wmax; i+=4;
   v[i+0] = Xmax; v[i+1] = Ymax; v[i+2] = Zmax;v[i+3]=Wmax; i+=4;
}

#ifdef __AVX__
/* Calculate bbox Subbox Coordinates */
static void
vbo_bbox_calc_subbox_coordinates(unsigned int vertSubBox,unsigned int vertCount,
                                 unsigned int first_idx,unsigned int second_idx,
                                 unsigned short* indices,float* vertices,
                                 unsigned int stride,
                                 struct vbo_bbox_cache_data *data)
{

   /* Retrieving the starting offset of the first and second subbox */
   unsigned int first = first_idx * vertSubBox;
   unsigned int second = second_idx * vertSubBox;

   float tmpVertexBuf[8] = {0.0};
   float tmp_buf_min[8] = {0.0};
   float tmp_buf_max[8] = {0.0};

   __m256 subMin = _mm256_set1_ps(FLT_MAX);
   __m256 subMax = _mm256_set1_ps(-FLT_MAX);

   /* Run both the subboxes for vertex count */
   for(unsigned int iter = 0; iter < vertCount; iter++){
      /* Calculate the vertex offset of first subbox */
      unsigned short index1 = indices[first+iter];
      /* Fetching the vertices for the first subbox */
      float* vertex1 = (float *)((char *)(vertices) + stride*index1);
      memcpy(tmpVertexBuf,vertex1, 3 * sizeof(float));

      /* Calculate the vertex offset of second subbox */
      unsigned short index2 = indices[second+iter];

      /* Fetching the vertices for the second subbox */
      float* vertex2 = (float *)((char *)(vertices) + stride*index2);
      memcpy(tmpVertexBuf+4,vertex2, 3 * sizeof(float));

      __m256 tmp = _mm256_loadu_ps(tmpVertexBuf);
      subMin = _mm256_min_ps(subMin, tmp);
      subMax = _mm256_max_ps(subMax, tmp);
   }

   /* compare full box values */
   fullMin = _mm256_min_ps(fullMin, subMin);
   fullMax = _mm256_max_ps(fullMax, subMax);

   /* store results */
   _mm256_storeu_ps(tmp_buf_min, subMin);
   _mm256_storeu_ps(tmp_buf_max, subMax);


   /* Update the min and max values in sub box coordinates for both
    * the subboxes
    */
   vbo_bbox_create_bounding_box(tmp_buf_min, tmp_buf_max,
              &(data->sub_bbox_array[first_idx].bounding_volume.vert_vec4[0]));
   vbo_bbox_create_bounding_box(tmp_buf_min+4, tmp_buf_max+4,
             &(data->sub_bbox_array[second_idx].bounding_volume.vert_vec4[0]));
}
#else
static void
vbo_bbox_calc_subbox_coordinates(
            unsigned int vertSubBox,
            unsigned int vertCount,
            unsigned int idx,
            unsigned short *indices,
            float *vertices,
            unsigned int stride,
            struct vbo_bbox_cache_data *data)
{
   /* Retrieving the starting offset of the first and second subbox */
   unsigned int first = idx * vertSubBox;

   float tmpVertexBuf[4] = {0.0};
   float tmp_buf_min[4] = {0.0};
   float tmp_buf_max[4] = {0.0};

   __m128 subMin = _mm_set1_ps(FLT_MAX);
   __m128 subMax = _mm_set1_ps(-FLT_MAX);

   /* Run both the subboxes for vertex count */
   for(unsigned int iter = 0; iter < vertCount; iter++){

      /* Calculate the vertex offset of first subbox */
      unsigned short index = indices[first+iter];
      /* Fetching the vertices for the first subbox */
      float* vertex = (float *)((char *)(vertices) + stride*index);
      memcpy(tmpVertexBuf,vertex, 3 * sizeof(float));

      __m128 tmp = _mm_loadu_ps(tmpVertexBuf);
      subMin = _mm_min_ps(subMin, tmp);
      subMax = _mm_max_ps(subMax, tmp);
   }

   /* compare full box values */
   fullMin = _mm_min_ps(fullMin, subMin);
   fullMax = _mm_max_ps(fullMax, subMax);

   /* store results */
   _mm_storeu_ps(tmp_buf_min, subMin);
   _mm_storeu_ps(tmp_buf_max, subMax);

   /* Update the min and max values in sub box coordinates for both
    * the subboxes
    */
   vbo_bbox_create_bounding_box(tmp_buf_min, tmp_buf_max,
                    &(data->sub_bbox_array[idx].bounding_volume.vert_vec4[0]));
}
#endif

/**
 *  Get pointer to VBO data.
 *  Pointer should be suitable for fast data reading, not data change.
 */
static
bool vbo_bbox_get_vbo_ptr(struct gl_context* gc, struct gl_buffer_object* vbo,
                          int offset, void** data, int* dataSize)
{
   assert(gc);
   assert(vbo);
   GLubyte* vboDataPtr = NULL;

   if (offset >= vbo->Size) {
      return false;
   }
   vboDataPtr = _mesa_MapNamedBuffer(vbo->Name,GL_WRITE_ONLY_ARB);
   if (vboDataPtr == NULL) {
      return false;
   }
   *data = vboDataPtr + offset;
   *dataSize = vbo->Size - offset;

   return true;
}

/**
 * Unlock VBO
 */
static inline
void vbo_bbox_release_vbo_ptr(struct gl_context* gc,
                              struct gl_buffer_object* vbo)
{
    assert(gc);
    assert(vbo);
    _mesa_UnmapNamedBuffer_no_error(vbo->Name);
}

/**
  * Check if given range of indices contains only degenerate triangles.
  */
static
bool vbo_bbox_is_degenerate(GLvoid *indices, GLuint count_in)
{
   assert(indices);
   assert(count_in % 3 == 0);

   GLuint triangle_count = count_in / 3;
   GLuint input_idx = 0;

   GLushort* ptr = (GLushort*)indices;
   for (GLuint i = 0; i < triangle_count; i++) {
      GLushort a = ptr[input_idx++];
      GLushort b = ptr[input_idx++];
      GLushort c = ptr[input_idx++];
      if (!(a == b || a == c || b == c)) {
          return false;
      }
   }
   return true;
}


/**
 * Calculate bounding boxes for given geometry.
 */
static
bool vbo_bbox_calculate_bounding_boxes_with_indices(struct gl_context *const gc,
                                             const vbo_bbox_cache_key *key,
                                             struct vbo_bbox_cache_data *data,
                                             void* indexData,
                                             int   indexDataSize)
{
   assert(gc);

   void* vertex_data = NULL;
   int vertex_datasize = 0;
   int vert_per_subBbox = mesa_bbox_env_variables.bbox_split_size;
   int sub_bbox_cnt = (key->count + vert_per_subBbox -1)/vert_per_subBbox;
   int *subbox_array;
   int idx =0;
   int non_degen_count = 0;

   assert(sub_bbox_cnt);

   struct gl_buffer_object* element_buffer = _mesa_lookup_bufferobj(gc,
                                                key->element_buf_name);

   struct gl_buffer_object * vertexattrib_buffer = _mesa_lookup_bufferobj(gc,
                                                       key->vertex_buf_name);

   if (element_buffer == NULL || vertexattrib_buffer == NULL) {
   return false;
   }

   if (!vbo_bbox_get_vbo_ptr(gc, vertexattrib_buffer,(int) key->offset,
                           &vertex_data, &vertex_datasize)) {
   return false;
   }

   assert(vertex_data);
   assert(vertex_datasize > 0);
   assert(indexData);
   assert(indexDataSize > 0);
   assert(key->indices_type == GL_UNSIGNED_SHORT);
   assert(indexDataSize > key->count * (int)key->type_size);

   /* Allocate memory for bounding boxes */
   if (!vbo_init_sub_bbox_array(sub_bbox_cnt,data)) {
   vbo_bbox_release_vbo_ptr(gc, vertexattrib_buffer);
   return false;
   }
   /* Initialize size of bounding boxes */
   for (int i = 0; i < sub_bbox_cnt; i++) {
   data->sub_bbox_array[i].vert_count = (i==sub_bbox_cnt-1)?
                            (key->count - i*vert_per_subBbox):vert_per_subBbox;
   data->sub_bbox_array[i].start_offset = i*vert_per_subBbox * key->type_size;
   }

   subbox_array = malloc(sub_bbox_cnt * sizeof(int));

   /* Check if all triangles withing bbox are degenerate (i.e triangles with
      zero area) */
   for (int i = 0; i < sub_bbox_cnt; i++) {
      GLubyte* ptr = (GLubyte *)indexData;
      data->sub_bbox_array[i].is_degenerate =
          vbo_bbox_is_degenerate((ptr + data->sub_bbox_array[i].start_offset),
                  data->sub_bbox_array[i].vert_count);

       if(!data->sub_bbox_array[i].is_degenerate)
       {
          subbox_array[idx++] = i;
          non_degen_count++;
       }
   }

   float tmp_buf_min[8] = {0.0};
   float tmp_buf_max[8] = {0.0};

#ifdef __AVX__
   int odd = non_degen_count % 2;
   int num_iter = non_degen_count/2;
   int iter;

   fullMin = _mm256_set1_ps(FLT_MAX);
   fullMax = _mm256_set1_ps(-FLT_MAX);
   idx = 0;
   for(iter = 0; iter < num_iter;iter++){
      idx = 2*iter;
      if(data->sub_bbox_array[subbox_array[idx]].vert_count ==
         data->sub_bbox_array[subbox_array[idx+1]].vert_count)
      {
         /* call the algorithm with the count */
         vbo_bbox_calc_subbox_coordinates(
                       vert_per_subBbox,
                       data->sub_bbox_array[subbox_array[idx]].vert_count,
                       subbox_array[idx],
                       subbox_array[idx+1],
                       (GLushort*)indexData,
                       (GLfloat*)vertex_data,
                       key->stride,
                       data
                     );
      }
      else
      {
         /* call the first one separately */
         vbo_bbox_calc_subbox_coordinates(vert_per_subBbox,
                            data->sub_bbox_array[subbox_array[idx]].vert_count,
                            subbox_array[idx],
                            subbox_array[idx],
                            (GLushort*)indexData,
                            (GLfloat*)vertex_data,
                            key->stride,
                            data);

         /* call the second one separately */
         vbo_bbox_calc_subbox_coordinates(vert_per_subBbox,
                          data->sub_bbox_array[subbox_array[idx+1]].vert_count,
                          subbox_array[idx+1],
                          subbox_array[idx+1],
                          (GLushort*)indexData,
                          (GLfloat*)vertex_data,
                          key->stride,
                          data);

      }

   }

   if(odd)
   {
      idx = 2*iter;
      /* call the last one separately */
      vbo_bbox_calc_subbox_coordinates(vert_per_subBbox,
                            data->sub_bbox_array[subbox_array[idx]].vert_count,
                            subbox_array[idx],
                            subbox_array[idx],
                            (GLushort*)indexData,
                            (GLfloat*)vertex_data,
                            key->stride,
                            data);
   }

   /* Finding the minimum from the full box 256 */
   __m128 firstlane   = _mm256_extractf128_ps(fullMin,0);
   __m128 secondlane = _mm256_extractf128_ps(fullMin,1);
   firstlane = _mm_min_ps(firstlane,secondlane);
   _mm_storeu_ps(tmp_buf_min,firstlane);

   /* Finding the maximum from the full box 256 */
    firstlane  = _mm256_extractf128_ps(fullMax,0);
    secondlane = _mm256_extractf128_ps(fullMax,1);
    firstlane = _mm_max_ps(firstlane,secondlane);
    _mm_storeu_ps(tmp_buf_max,firstlane);

#else
    fullMin = _mm_set1_ps(FLT_MAX);
    fullMax = _mm_set1_ps(-FLT_MAX);

    for(unsigned int i=0; i< non_degen_count; i++){
       //call the algorithm with the count
       vbo_bbox_calc_subbox_coordinates(vert_per_subBbox,
                              data->sub_bbox_array[subbox_array[i]].vert_count,
                              subbox_array[i],
                              (GLushort*)indexData,
                              (GLfloat*)vertex_data,
                              key->stride,
                              data);
       }
       _mm_storeu_ps(tmp_buf_min, fullMin);
       _mm_storeu_ps(tmp_buf_max, fullMax);
#endif

    /* Set up bounding box as 8 vertices and store in bbox data */
    vbo_bbox_create_bounding_box(tmp_buf_min, tmp_buf_max,
                          &(data->full_box.bounding_volume.vert_vec4[0]));

    if(subbox_array)
      free(subbox_array);
    vbo_bbox_release_vbo_ptr(gc, vertexattrib_buffer);
    data->valid = true;

    return true;
}


/**
  * Calculate bounding boxes for given geometry.
  */
static inline
bool vbo_bbox_calculate_bounding_boxes(struct gl_context *const gc,
                                       const vbo_bbox_cache_key *key,
                                       struct vbo_bbox_cache_data* data)
{
    assert(gc);
    assert(key->indices_type == GL_UNSIGNED_SHORT);

    void* pIndexData     = NULL;
    int   indexDataSize  = 0;

    struct gl_buffer_object* element_buffer = _mesa_lookup_bufferobj(gc,
                                                        key->element_buf_name);
    if (element_buffer == NULL) {
      return false;
    }
    if (!vbo_bbox_get_vbo_ptr(gc, element_buffer, (int) key->indices,
                              &pIndexData, &indexDataSize)) {
      return false;
    }

    bool ret = vbo_bbox_calculate_bounding_boxes_with_indices(gc, key, data,
                                                    pIndexData, indexDataSize);

    vbo_bbox_release_vbo_ptr(gc, element_buffer);

    return ret;
}


/**
  *  Create new bounding box cache entry
  */
static
struct vbo_bbox_cache_data* vbo_bbox_create_data(struct gl_context *const gc,
                                                 const vbo_bbox_cache_key *key)
{
   assert(gc);

   struct gl_buffer_object* element_buffer = vbo_get_current_element_buffer(gc);
   struct gl_buffer_object* vertexattrib_buffer =
                                             vbo_get_current_vertex_buffer(gc);

   if ((vertexattrib_buffer == NULL) ||
     (element_buffer == NULL)){
      return NULL;
   }

   mesa_bbox_opt * BboxOpt = gc->Pipeline.BboxOpt;
   assert(BboxOpt);

   struct vbo_bbox_cache_data* data = (struct vbo_bbox_cache_data *) malloc(
                                      sizeof (struct vbo_bbox_cache_data));
   data->full_box.is_degenerate = false;

   if (data == NULL){
      return NULL;
   }
   /* Initialize the cache data and variables in cache data */
   data->valid = false;
   data->need_new_calculation = true;
   data->init_delaycnt = 0;
   data->init_delaylimit = 0;
   data->vertpos_vbo_changecnt = 0;
   data->indices_vbo_changecnt = 0;
   data->sub_bbox_cnt = 0;
   data->sub_bbox_array = NULL;
   data->drawcnt_bbox_helped = 0;
   data->last_use_frame = 0;
   data->vertpos_vbo_changecnt   = vertexattrib_buffer->data_change_counter;
   data->indices_vbo_changecnt   = element_buffer->data_change_counter;

   /* This defines for how many cache hits we wait before actually creating the
    * data */
   data->init_delaylimit = vbo_bbox_get_delay(BboxOpt);

   /* At this point data is not valid yet */
   _mesa_bbox_cache_insert(gc,gc->Pipeline.BboxOpt->cache,
                           key,sizeof(vbo_bbox_cache_key),data);

   return data;
}


/**
 * Check if contents of the VBO buffers have changed since data entry
 * was created.
 */
static inline
bool vbo_bbox_validate_data(
    struct gl_context *const gc,
    const vbo_bbox_cache_key *key,
    struct vbo_bbox_cache_data* data)
{
   assert(gc);

   struct gl_buffer_object* element_buffer = _mesa_lookup_bufferobj(gc,
                                               key->element_buf_name);

   struct gl_buffer_object * vertexattrib_buffer = _mesa_lookup_bufferobj(gc,
                                                      key->vertex_buf_name);

   if (element_buffer == NULL || vertexattrib_buffer == NULL) {
      return false;
   }

   if ((element_buffer->data_change_counter != data->indices_vbo_changecnt) ||
     (vertexattrib_buffer->data_change_counter != data->vertpos_vbo_changecnt)
     ) {
      return false;
   }
   return true;
}

/**
 *  Retrieve bounding box data from cache.
 */
static inline
struct vbo_bbox_cache_data* vbo_bbox_get_bounding_boxes(
                                                 struct gl_context *const gc,
                                                 const vbo_bbox_cache_key *key)
 {

   assert(gc);
   mesa_bbox_opt * BboxOpt = gc->Pipeline.BboxOpt;
   assert(BboxOpt);
   struct vbo_bbox_cache_data* data = _mesa_search_bbox_cache(BboxOpt->cache,
                                              key, sizeof(vbo_bbox_cache_key));
   if (data) {
      if (data->need_new_calculation == false)
      {
         /* Data is initialized and valid */
         if (data->valid) {
             if (vbo_bbox_validate_data(gc, key, data)) {
                data->mvp_valid = true;
                return data;
             }
         }
         else {
             data->valid = false;
             return NULL;
         }
      }
   }
   else {
      /* Data does not exist, create it */
      data = vbo_bbox_create_data(gc, key);
      if (data == NULL)
      {
         return NULL;
      }
   }
   if ((data->need_new_calculation) &&
     (data->init_delaycnt++ >= data->init_delaylimit)) {
     data->valid = false;
     data->need_new_calculation = false;
     data->mvp_valid = false;

     if (!vbo_bbox_validate_data(gc, key, data)) {
         struct gl_buffer_object * element_buffer =
                 _mesa_lookup_bufferobj(gc,key->element_buf_name);

         struct gl_buffer_object * vertexattrib_buffer =
                 _mesa_lookup_bufferobj(gc,key->vertex_buf_name);

         if ((vertexattrib_buffer == NULL) ||
             (element_buffer == NULL)) {
             return NULL;
         }
         data->vertpos_vbo_changecnt = vertexattrib_buffer->data_change_counter;
         data->indices_vbo_changecnt = element_buffer->data_change_counter;
      }
      if (gc->volume_type == BOUNDING_VOLUME_AABB) {
         /* Calculate bounding boxes */
         if (vbo_bbox_calculate_bounding_boxes(gc, key, data)) {
             return data;
         }
      }
   }
   return NULL;
}

/**
 * This function is called when we updating the element buffer. because the
 * element-buffer has changed we have to update the relevant bbox data:
 */
void vbo_bbox_element_buffer_update(struct gl_context *const gc,
                                    struct gl_buffer_object *buffer,
                                    const void* data,
                                    int offset,
                                    int size)
{
   mesa_bbox_opt * BboxOpt = gc->Pipeline.BboxOpt;

   if (BboxOpt) {
      struct gl_segment updateSegment;
      updateSegment.Left = offset;
      updateSegment.Right = offset+size;

      mesa_bbox_cache * buffer_map = BboxOpt->cache;
      if (buffer_map) {
         struct vbo_bbox_cache_data *c;
         GLuint i = 0;
         for (i = 0; i < buffer_map->size; i++)
            for (c = buffer_map->items[i]; c != NULL ; c = c->next) {
               struct vbo_bbox_cache_data * bbox_data = buffer_map->items[i];
               struct gl_buffer_object *buffObj = _mesa_lookup_bufferobj(gc,
                                            bbox_data->key->element_buf_name);

               if(buffObj != buffer)
               continue;

               if (!bbox_data) {

                  assert(bbox_data);
                  return;
               }

               bbox_data->indices_vbo_changecnt = buffer->data_change_counter;

               struct gl_segment element_bufferSegment;
               element_bufferSegment.Left = bbox_data->key->indices;
               element_bufferSegment.Right = bbox_data->key->indices +
                           (bbox_data->key->count * bbox_data->key->type_size);

               if (bbox_data->valid &&
                   intersect(&element_bufferSegment, &updateSegment)) {
                  bbox_data->valid = false;
                  bbox_data->need_new_calculation = true;

                  if (superset(&element_bufferSegment,&updateSegment))
                  {
                      int offset_in_newdata = bbox_data->key->indices - offset;
                      int newsize         = size - offset_in_newdata;

                      assert(offset_in_newdata >= 0);
                      assert(newsize > 0);

                      GLchar * start_data = (GLchar *)data + offset_in_newdata;

                   if (vbo_bbox_calculate_bounding_boxes_with_indices(gc,
                       bbox_data->key, bbox_data, start_data, newsize)) {
                       bbox_data->need_new_calculation = false;
                       bbox_data->mvp_valid = false;
                   }
               }
            }
         }
      }
   }
}

/**
 *  Generate 6 clip planes from MVP.
 */
static
void vbo_bbox_get_frustum_from_mvp(vbo_bbox_frustum *frustum, GLmatrix* mvpin)
{

   GLfloat in_mat[4][4] = {0};
   {
   #define M(row,col)  m[col*4+row]
      in_mat[0][0] = mvpin->M(0,0);
      in_mat[0][1] = mvpin->M(0,1);
      in_mat[0][2] = mvpin->M(0,2);
      in_mat[0][3] = mvpin->M(0,3);

      in_mat[1][0] = mvpin->M(1,0);
      in_mat[1][1] = mvpin->M(1,1);
      in_mat[1][2] = mvpin->M(1,2);
      in_mat[1][3] = mvpin->M(1,3);

      in_mat[2][0] = mvpin->M(2,0);
      in_mat[2][1] = mvpin->M(2,1);
      in_mat[2][2] = mvpin->M(2,2);
      in_mat[2][3] = mvpin->M(2,3);

      in_mat[3][0] = mvpin->M(3,0);
      in_mat[3][1] = mvpin->M(3,1);
      in_mat[3][2] = mvpin->M(3,2);
      in_mat[3][3] = mvpin->M(3,3);
   #undef M
   }

   /* Frustum plane calculation */

   /* Left plane */
   frustum->plane[0].a = in_mat[3][0] + in_mat[0][0];
   frustum->plane[0].b = in_mat[3][1] + in_mat[0][1];
   frustum->plane[0].c = in_mat[3][2] + in_mat[0][2];
   frustum->plane[0].d = in_mat[3][3] + in_mat[0][3];

   /* Right plane */
   frustum->plane[1].a = in_mat[3][0] - in_mat[0][0];
   frustum->plane[1].b = in_mat[3][1] - in_mat[0][1];
   frustum->plane[1].c = in_mat[3][2] - in_mat[0][2];
   frustum->plane[1].d = in_mat[3][3] - in_mat[0][3];

   /* Top plane */
   frustum->plane[2].a = in_mat[3][0] - in_mat[1][0];
   frustum->plane[2].b = in_mat[3][1] - in_mat[1][1];
   frustum->plane[2].c = in_mat[3][2] - in_mat[1][2];
   frustum->plane[2].d = in_mat[3][3] - in_mat[1][3];

   /* Bottom plane */
   frustum->plane[3].a = in_mat[3][0] + in_mat[1][0];
   frustum->plane[3].b = in_mat[3][1] + in_mat[1][1];
   frustum->plane[3].c = in_mat[3][2] + in_mat[1][2];
   frustum->plane[3].d = in_mat[3][3] + in_mat[1][3];

   /* Far plane */
   frustum->plane[4].a = in_mat[3][0] - in_mat[2][0];
   frustum->plane[4].b = in_mat[3][1] - in_mat[2][1];
   frustum->plane[4].c = in_mat[3][2] - in_mat[2][2];
   frustum->plane[4].d = in_mat[3][3] - in_mat[2][3];

   /* Near plane */
   frustum->plane[5].a = in_mat[3][0] + in_mat[2][0];
   frustum->plane[5].b = in_mat[3][1] + in_mat[2][1];
   frustum->plane[5].c = in_mat[3][2] + in_mat[2][2];
   frustum->plane[5].d = in_mat[3][3] + in_mat[2][3];

    /* Calculate octants */
    for(int n = 0; n < 6; ++n) {
        frustum->octant[n] = (frustum->plane[n].a >=0 ? 1 : 0) |
                             (frustum->plane[n].b >=0 ? 2 : 0) |
                             (frustum->plane[n].c >=0 ? 4 : 0);
        normalize(&(frustum->plane[n]));
    }
}


/**
 * Calculate distance form a point to place
 *
 */
static inline
float vbo_bbox_dist_from_point_to_plane(
    const vbo_bbox_frustum_plane *plane,
    const vbo_vec4f *point)
{
    return (plane->a * point->x + plane->b * point->y + plane->c *
            point->z + plane->d);
}


/**
 * Description:
 * Bounding box clipping algorthm
 * BBOX_CLIP_INSIDE - bounding box is fully inside frustum
 * BBOX_CLIP_OUTSIDE - bounding box is fully outside frustum
 * BBOX_CLIP_INTERSECT - bounding box intersects with frustum
 */
static
enum vbo_bbox_clip_result vbo_bbox_fast_clipping_test(bounding_info* bbox,
                                               const vbo_bbox_frustum *frustum)
{
   assert(bbox);
   vbo_vec4f *aabb = bbox->bounding_volume.vert_vec4;

   enum vbo_bbox_clip_result result = BBOX_CLIP_INSIDE;
   for (int i = 0; i < 6; ++i)
   {
      unsigned char normalOctant = frustum->octant[i];

      /* Test near and far vertices of AABB according to plane normal.
       * Plane equation can be normalized to save some divisions.
       */
      float farDistance = vbo_bbox_dist_from_point_to_plane(
                                                        &(frustum->plane[i]),
                                                        &(aabb[normalOctant]));
      if (farDistance < 0.0f) {
         return BBOX_CLIP_OUTSIDE;
      }

      float nearDistance = vbo_bbox_dist_from_point_to_plane(
                                                    &(frustum->plane[i]),
                                                    &(aabb[normalOctant ^ 7]));
      if (nearDistance < 0.0f) {
         result = BBOX_CLIP_INTERSECT;
      }
   }

   return result;
}

/**
 *  Wrapper for clip algorithm.
 */
static inline
enum vbo_bbox_clip_result vbo_bbox_clip(bounding_info* bbox,
                                        const vbo_bbox_frustum *frustum)
{
   assert(bbox);
   if (bbox->is_degenerate) {
      return BBOX_CLIP_DEGEN;
   }
   return vbo_bbox_fast_clipping_test(bbox, frustum);
}

/**
 * Bounding box drawelements implementation
 */
void
vbo_bbox_drawelements(struct gl_context *ctx, GLenum mode,
                      GLboolean index_bounds_valid, GLuint start, GLuint end,
                      GLsizei count, GLenum type, const GLvoid * indices,
                      GLint basevertex, GLuint numInstances,
                      GLuint baseInstance)
{
   assert(ctx);
   GLuint type_size;
   /* BOUNDING VOLUME: Checks would remain same I guess */
   bool draw_call_supported = vbo_bbox_check_supported_draw_call(ctx,
                                                               mode,
                                                               count,
                                                               type,
                                                               indices,
                                                               basevertex);
   if (!draw_call_supported ||
   (mesa_bbox_env_variables.bbox_enable < MESA_BBOX_ENABLE_FORCE_CLIPPING)) {
      MESA_BBOX("Aborting MESA_BBOX : BBOX Is not ENABLED !!! \n");
      vbo_validated_drawrangeelements(ctx, mode, index_bounds_valid, start,
          end, count, type, indices, basevertex, numInstances, baseInstance);
      return;
   }

   type_size = sizeof(GLushort);
   vbo_bbox_cache_key key;/* Need not initialize key its done in prepare key */
   vbo_bbox_prepare_key(ctx, mode, count, type, type_size, indices, basevertex,
                        &key);
   /*BOUNDING VOLUME: Call bounding volume creation based on bounding volume
    * type */

   struct vbo_bbox_cache_data* cached_bbox = vbo_bbox_get_bounding_boxes(ctx,
                                                                         &key);

   if (cached_bbox == NULL) {
      MESA_BBOX("Aborting MESA_BBOX : New Object not in Cache!!! \n");
      vbo_validated_drawrangeelements(ctx, mode, index_bounds_valid, start, end,
                                      count, type, indices, basevertex,
                                      numInstances, baseInstance);
      return;
   }

   GLint loc = vbo_bbox_get_mvp(ctx);
   if (loc < 0) {
     MESA_BBOX("MVP Location Error\n");
     /* TBD: Free cache here */
     vbo_validated_drawrangeelements(ctx, mode, index_bounds_valid, start,
                                     end, count, type, indices, basevertex,
                                     numInstances, baseInstance);
     return;
   }

   GLfloat *mvp_ptr = (GLfloat *)
                ctx->_Shader->ActiveProgram->data->UniformStorage[loc].storage;
   vbo_bbox_frustum frustum;
   bool recalculate_subbox_clip = false;

   if(cached_bbox->mvp_valid == false ||
      memcmp(cached_bbox->mvp, mvp_ptr,16*sizeof(GLfloat))) {
      memcpy(&(cached_bbox->mvp), mvp_ptr, 16*sizeof(GLfloat));
      cached_bbox->mvpin.m = cached_bbox->mvp;
      vbo_bbox_get_frustum_from_mvp(&frustum,&(cached_bbox->mvpin));

      /* BOUNDING VOLUME: Call specific function to calculate clip results */

      if (ctx->volume_type == BOUNDING_VOLUME_AABB) {
          cached_bbox->full_box.clip_result =
                  vbo_bbox_clip(&(cached_bbox->full_box), &frustum);
      }
      recalculate_subbox_clip = true;
   }

    /* Calculate frustum planes */
    MESA_BBOX("MESA_BBOX: Full Box ClipResult:%d \n",
                    cached_bbox->full_box.clip_result);
    switch (cached_bbox->full_box.clip_result) {
      case BBOX_CLIP_OUTSIDE:
         /* Geometry outside view frustum, dont draw it */
        return;
      case BBOX_CLIP_INSIDE:
        vbo_validated_drawrangeelements(ctx, mode, index_bounds_valid, start,
                                        end, count, type, indices, basevertex,
                                        numInstances, baseInstance);
        return;
      /* case BBOX_CLIP_INTERSECT: */
      default:
        MESA_BBOX("MESA_BBOX: Vertices INTERSECTING with the frustum, going"
                  " for Sub Bboxes: \n");
        break;
    }

    GLsizei count_to_send = 0;
    GLsizei count_to_drop = 0;
    GLvoid* offset_to_send = NULL;
    bool clipped = false;
    unsigned potential_clipped = 0;

    for (int i = 0; i < cached_bbox->sub_bbox_cnt; i++) {
      int new_count = cached_bbox->sub_bbox_array[i].vert_count;

      if(recalculate_subbox_clip) {
         if (ctx->volume_type == BOUNDING_VOLUME_AABB) {
             cached_bbox->sub_bbox_array[i].clip_result =
                 vbo_bbox_clip(&(cached_bbox->sub_bbox_array[i]), &frustum);
         }
      }

      switch (cached_bbox->sub_bbox_array[i].clip_result)
      {
      case BBOX_CLIP_OUTSIDE:
         count_to_drop += new_count;
         potential_clipped += new_count;

         break;
      case BBOX_CLIP_DEGEN:
         count_to_drop += new_count;
         potential_clipped += new_count;
         break;
        default:
            /* Sub bounding box intersects with view, draw/save it
             * for later draw
             */
            if (count_to_send == 0) {
                /* Starting new batch */
                count_to_send = new_count;
                offset_to_send = (char*)indices +
                             cached_bbox->sub_bbox_array[i].start_offset;
            }
            else {
                if (count_to_drop >=
                   (int)(mesa_bbox_env_variables.bbox_split_size *
                   BBOX_MIN_SPLITTED_DRAW_TO_DROP)) {

                    /* Draw accumulated geometry */
                    vbo_validated_drawrangeelements(ctx, mode,
                       index_bounds_valid, start, end, count_to_send, type,
                       offset_to_send, basevertex, numInstances, baseInstance);

                    /* Reset accumulated draws */
                    count_to_send = 0;
                    offset_to_send = (char*)indices +
                             cached_bbox->sub_bbox_array[i].start_offset;

                    clipped = true;
                }
                else
                {
                    count_to_send += count_to_drop;
                }

                /* append to current batch of sent primitives */
                count_to_send += new_count;
            }

            count_to_drop = 0;
            break;
        }
    }

    if (count_to_send > 0)
    {
        vbo_validated_drawrangeelements(ctx, mode, index_bounds_valid, start,
                                        end, count_to_send, type,
                                        offset_to_send, basevertex,
                                        numInstances, baseInstance);
    }


    clipped |= (count_to_drop >= (int)(mesa_bbox_env_variables.bbox_split_size *
                                       BBOX_MIN_SPLITTED_DRAW_TO_DROP));

    if (clipped)
    {
        cached_bbox->drawcnt_bbox_helped =
                                      MIN(cached_bbox->drawcnt_bbox_helped + 1,
                                      BBOX_MAX_VAL_FOR_EFFECTIVE_DRAWS_COUNTER);
    }
    else
    {
        if (potential_clipped == 0)
        {
            cached_bbox->drawcnt_bbox_helped =
                                      MAX(cached_bbox->drawcnt_bbox_helped - 1,
                                      BBOX_MIN_VAL_FOR_EFFECTIVE_DRAWS_COUNTER);
        }
    }
}

/**
 * Initialization for bounding box optimization
 */
void vbo_bbox_init(struct gl_context* const gc)
{
   assert(gc);

   const char * mesa_bbox_opt = getenv("MESA_BBOX_OPT_ENABLE");
   if (mesa_bbox_opt !=NULL)
      mesa_bbox_env_variables.bbox_enable = atoi(mesa_bbox_opt);

   mesa_bbox_opt = getenv("MESA_OPT_SPLIT_SIZE");
   if (mesa_bbox_opt !=NULL)
      mesa_bbox_env_variables.bbox_split_size = atoi(mesa_bbox_opt);

   mesa_bbox_opt = getenv("MESA_BBOX_MIN_VERTEX_CNT");
   if (mesa_bbox_opt != NULL)
      mesa_bbox_env_variables.bbox_min_vrtx_count = atoi(mesa_bbox_opt);

   if (!gc->Const.EnableBoundingBoxCulling) {
      mesa_bbox_env_variables.bbox_enable = MESA_BBOX_ENABLE_OFF;
      mesa_bbox_env_variables.bbox_split_size = 0x7fffffff;
      mesa_bbox_env_variables.bbox_min_vrtx_count = 0;
   }

   mesa_bbox_opt = getenv("MESA_OPT_TRACE_LEVEL");
   if (mesa_bbox_opt !=NULL)
      mesa_bbox_env_variables.bbox_trace_level = atoi(mesa_bbox_opt);

   if (mesa_bbox_env_variables.bbox_enable == MESA_BBOX_ENABLE_AUTO) {
      /* Android/Linux: enable of Gen7.5, Gen8 and Gen9 */
      #if (IGFX_GEN == IGFX_GEN9)
      mesa_bbox_env_variables.bbox_enable = MESA_BBOX_ENABLE_SMART;
      #else
      mesa_bbox_env_variables.bbox_enable = MESA_BBOX_ENABLE_OFF;
      #endif
      if (mesa_bbox_env_variables.bbox_enable == MESA_BBOX_ENABLE_SMART && 1)
      {
         mesa_bbox_env_variables.bbox_enable =
                                             MESA_BBOX_ENABLE_FORCE_CLIPPING;
      }
   }
   /* BOUNDING VOLUME: Add initializations based on bounding volume here */
   switch (gc->volume_type) {
           case BOUNDING_VOLUME_AABB:
                   if (mesa_bbox_env_variables.bbox_enable) {
                      mesa_bbox_env_variables.bbox_split_size =
                             (mesa_bbox_env_variables.bbox_split_size / 3) * 3;
                      mesa_bbox_env_variables.bbox_min_vrtx_count =
                         (mesa_bbox_env_variables.bbox_min_vrtx_count / 3) * 3;

                      assert(gc->Pipeline.BboxOpt == NULL);

                      gc->Pipeline.BboxOpt = malloc(sizeof(mesa_bbox_opt));
                      if (!gc->Pipeline.BboxOpt) {
                          /* No memory, disable bbox optimization */
                          mesa_bbox_env_variables.bbox_enable =
                                  MESA_BBOX_ENABLE_OFF;
                          mesa_bbox_env_variables.bbox_split_size = 0x7fffffff;
                          mesa_bbox_env_variables.bbox_min_vrtx_count = 0;
                          return;
                      }
                      else {
                          gc->Pipeline.BboxOpt->calc_delay = BBOX_CALC_MIN_DELAY;
                          gc->Pipeline.BboxOpt->cache = _mesa_new_bbox_cache();

                          if (!gc->Pipeline.BboxOpt->cache) {
                              free(gc->Pipeline.BboxOpt);
                              MESA_BBOX("MESA_BBOX: Cache creation failed\n");
                              return;
                           }
                      }
                      if (mesa_bbox_env_variables.bbox_trace_level > 0) {
                          MESA_BBOX("\nMESA BBOX OPT config: \
                               bboxOptEnable = %d, bboxOptMinVertexCount = %d,\
                               bboxOptSplitSize = %d\n",
                                   mesa_bbox_env_variables.bbox_enable,
                                   mesa_bbox_env_variables.bbox_min_vrtx_count,
                                   mesa_bbox_env_variables.bbox_split_size);
                           }
                   }
                   /* Initialize some function pointers so that we dont have to
                    * check bounding volume type for every draw call */
                   break;
           case BOUNDING_VOLUME_OBB:
                   /* Init for OBB bounding volume */
                   break;
           case BOUNDING_VOLUME_SPHERE:
                   /* Init for SPHERE bounding volume */
                   break;
           case BOUNDING_VOLUME_DOP:
                   /* Init for DOP bounding volume */
                   break;
           default:
                   MESA_BBOX("BOUNDING VOLUME TYPE IS INCORRECT\n");
                   break;
   }
}


/**
 *  Free resources associated with bounding box optimization.
 *  To be called when context is destroyed
 */
void vbo_bbox_free(struct gl_context* const gc)
{
   assert(gc);

   if (gc->Pipeline.BboxOpt) {

      if(gc->Pipeline.BboxOpt->cache)
      {
	_mesa_delete_bbox_cache(gc,gc->Pipeline.BboxOpt->cache);
      	gc->Pipeline.BboxOpt->cache = NULL;
      }

      if(gc->Pipeline.BboxOpt) {
      	free(gc->Pipeline.BboxOpt);
      	gc->Pipeline.BboxOpt = NULL;
      }
   }
}
