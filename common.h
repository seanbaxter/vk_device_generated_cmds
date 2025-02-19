/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef CSFTHREADED_COMMON_H
#define CSFTHREADED_COMMON_H

#define VERTEX_POS_OCTNORMAL      0

// changing these orders may break a lot of things ;)
#define DRAW_UBO_SCENE     0
#define DRAW_UBO_MATRIX    1
#define DRAW_UBO_MATERIAL  2

#define ANIM_UBO              0
#define ANIM_SSBO_MATRIXOUT   1
#define ANIM_SSBO_MATRIXORIG  2

#define ANIMATION_WORKGROUPSIZE 256

#ifndef SHADER_PERMUTATION
#define SHADER_PERMUTATION 1
#endif

//////////////////////////////////////////////////////////////////////////

// see resources_vk.hpp

#ifndef UNIFORMS_MULTISETSDYNAMIC
#define UNIFORMS_MULTISETSDYNAMIC 0
#endif
#ifndef UNIFORMS_PUSHCONSTANTS_ADDRESS
#define UNIFORMS_PUSHCONSTANTS_ADDRESS 1
#endif
#ifndef UNIFORMS_TECHNIQUE
#define UNIFORMS_TECHNIQUE UNIFORMS_PUSHCONSTANTS_ADDRESS
#endif

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus

enum BindingMode
{
  BINDINGMODE_DSETS,
  BINDINGMODE_PUSHADDRESS,
  NUM_BINDINGMODES,
};

struct shaders_t {
  const char* spirv_data;
  size_t spirv_size;

  const char* comp_animation;
  const char* vert[2];
  const char* frag[2][128];
};

const extern shaders_t shaders;

namespace generatedcmds {
  using namespace nvmath;
#endif

struct SceneData {
  mat4  viewProjMatrix;
  mat4  viewMatrix;
  mat4  viewMatrixIT;

  vec4  viewPos;
  vec4  viewDir;
  
  vec4  wLightPos;
  
  ivec2 viewport;
  ivec2 _pad;
};

// must match cadscene
struct MatrixData {
  mat4 worldMatrix;
  mat4 worldMatrixIT;
  mat4 objectMatrix;
  mat4 objectMatrixIT;
};

// must match cadscene
struct MaterialSide {
  vec4 ambient;
  vec4 diffuse;
  vec4 specular;
  vec4 emissive;
};

struct MaterialData {
  MaterialSide sides[2];
};

struct AnimationData {
  uint    numMatrices;
  float   time;
  vec2   _pad0;

  vec3    sceneCenter;
  float   sceneDimension;
};

#ifdef __cplusplus
}
#endif


#endif
