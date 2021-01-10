namespace nvmath { }
#include "common.h"

using namespace generatedcmds;

template<typename type_t, int location>
[[using spirv: in, location(location)]]
type_t shader_in;

template<typename type_t, int location>
[[using spirv: out, location(location)]]
type_t shader_out;

template<typename type_t, int binding, int set = 0>
[[using spirv: buffer, binding(binding), set(set)]]
type_t shader_buffer;

template<typename type_t, int binding, int set = 0>
[[using spirv: uniform, binding(binding), set(set)]]
type_t shader_uniform;

[[spirv::push]]
struct {
  const MatrixData* matrixData;
  const MaterialData* materialData;
} push;

struct interpolants_t {
  vec3 wPos;
  vec3 wNormal;
  vec3 oNormal;
};

// oct functions from http://jcgt.org/published/0003/02/01/paper.pdf
inline vec2 oct_signNotZero(vec2 v) {
  return vec2((v.x >= 0) ? +1 : -1, (v.y >= 0) ? +1 : -1);
}
inline vec3 oct_to_float32x3(vec2 e) {
  vec3 v = vec3(e.xy, 1 - abs(e.x) - abs(e.y));
  if (v.z < 0) v.xy = (1 - abs(v.yx)) * oct_signNotZero(v.xy);
  return normalize(v);
}
inline vec2 float32x3_to_oct(vec3 v) {
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (v.z <= 0) ? ((1 - abs(p.yx)) * oct_signNotZero(p)) : p;
}

template<BindingMode bindingMode>
[[spirv::vert]]
void vert_shader() {
  vec4 inPosNormal = shader_in<vec4, VERTEX_POS_OCTNORMAL>;

  vec3 inNormal = oct_to_float32x3(unpackSnorm2x16(floatBitsToUint(inPosNormal.w)));

  mat4 worldMatrix, worldMatrixIT, viewMatrix;
  if constexpr(BINDINGMODE_PUSHADDRESS == bindingMode) {
    worldMatrix   = push.matrixData->worldMatrix;
    worldMatrixIT = push.matrixData->worldMatrixIT;
    viewMatrix    = shader_uniform<SceneData, DRAW_UBO_SCENE, 0>.viewProjMatrix;

  } else {
    worldMatrix   = shader_uniform<MatrixData, 0, DRAW_UBO_MATRIX>.worldMatrix;
    worldMatrixIT = shader_uniform<MatrixData, 0, DRAW_UBO_MATRIX>.worldMatrixIT;
    viewMatrix    = shader_uniform<SceneData, 0, DRAW_UBO_SCENE>.viewProjMatrix;
  }

  vec3 wPos     = (worldMatrix   * vec4(inPosNormal.xyz, 1)).xyz;
  vec3 wNormal  = mat3(worldMatrixIT) * inNormal;

  mat4 viewProjMatrix = shader_uniform<SceneData, 0>.viewProjMatrix;
  glvert_Output.Position = viewMatrix * vec4(wPos, 1);

  shader_out<interpolants_t, 0> = {
    wPos, wNormal, inNormal, 
  };
}

template<BindingMode bindingMode, int permutation>
[[spirv::frag]]
void frag_shader() {
  MaterialSide side;
  SceneData scene;
  mat4 viewMatrix;

  if constexpr(BINDINGMODE_PUSHADDRESS == bindingMode) {
    side  = push.materialData->sides[glfrag_FrontFacing];
    scene = shader_uniform<SceneData, DRAW_UBO_SCENE, 0>;

  } else {
    side  = shader_uniform<MaterialData, 0, DRAW_UBO_MATERIAL>.sides[glfrag_FrontFacing];
    scene = shader_uniform<SceneData, 0, DRAW_UBO_SCENE>;
  }

  // Apply shading.
  vec4 color = side.ambient + side.emissive;

  // Apply stipple. 
  ivec2 pixel = ivec2(glfrag_FragCoord.xy);
  pixel /= (permutation % 8) + 1;
  pixel %= (permutation % 2) + 1;
  pixel = 1 - pixel;

  interpolants_t in = shader_in<interpolants_t, 0>;

  color = mix(color, vec4(.5f * in.oNormal + .5f, 1), .5f * pixel.x * pixel.y);
  color += .001f * permutation;

  vec3 eyePos(
    scene.viewMatrixIT[0].w, 
    scene.viewMatrixIT[1].w, 
    scene.viewMatrixIT[2].w
  );

  vec3 lightDir = normalize(scene.wLightPos.xyz - in.wPos);
  vec3 viewDir  = normalize(eyePos - in.wPos);
  vec3 halfDir  = normalize(lightDir + viewDir);
  vec3 normal   = normalize(in.wNormal) * (glfrag_FrontFacing ? 1 : -1);

  float ldot = dot(normal, lightDir);
  normal *= sign(ldot);
  ldot   *= sign(ldot);

  color += side.diffuse * ldot;
  color += side.specular * pow(max(0.f, dot(normal,halfDir)), 16.f);
  
  shader_out<vec4, 0> = color;
}


////////////////////////////////////////////////////////////////////////////////

[[using spirv: uniform, binding(ANIM_UBO)]]
AnimationData anim;

[[using spirv: buffer, writeonly, binding(ANIM_SSBO_MATRIXOUT)]]
MatrixData animated[];

[[using spirv: buffer, readonly, binding(ANIM_SSBO_MATRIXORIG)]]
MatrixData original[];

[[using spirv: comp, local_size(ANIMATION_WORKGROUPSIZE)]]
void comp_animation() {
  int gid = glcomp_GlobalInvocationID.x;
  if(gid >= anim.numMatrices)
    return;

  float s = 1 - (float)gid / anim.numMatrices;
  float movement = 4;             // time until all objects done with moving (<= sequence*0.5)
  float sequence = movement * 2 + 3;  // time for sequence
  
  float timeS = fract(anim.time / sequence) * sequence;
  float time  = 
    clamp(timeS -      s  * movement,                   0.f, 1.f) - 
    clamp(timeS - (1 - s) * movement - sequence * 0.5f, 0.f, 1.f);
  
  float scale = smoothstep(0,1,time);
  
  mat4 matrixOrig = original[gid].worldMatrix;
  vec3 pos  = matrixOrig[3].xyz;
  vec3 away = (pos - anim.sceneCenter);
  
  int diridx = gid % 3;
  int sidx   = gid % 6;

  vec3 delta(
    diridx == 0,
    diridx == 1,
    diridx == 2
  );

  delta *= -sign(sidx - 2.5f);
  delta *= sign(dot(away, delta));
  
  delta = normalize(delta);
  pos += delta * scale * anim.sceneDimension;
  
  animated[gid].worldMatrix = mat4(
    matrixOrig[0],
    matrixOrig[1], 
    matrixOrig[2], 
    vec4(pos, 1)
  );
}

const shaders_t shaders {
  __spirv_data,
  __spirv_size,

  @spirv(comp_animation),
  @spirv(vert_shader<BINDINGMODE_DSETS>),
  @spirv(vert_shader<BINDINGMODE_PUSHADDRESS>),
  @spirv(frag_shader<BINDINGMODE_DSETS, __integer_pack(128)>)...,
  @spirv(frag_shader<BINDINGMODE_PUSHADDRESS, __integer_pack(128)>)...
};