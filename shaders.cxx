namespace nvmath { }
#include "common.h"

using namespace generatedcmds;

template<typename type_t, int location>
[[using spirv: in, location(location)]]
type_t shader_in;

template<typename type_t, int location>
[[using spirv: out, location(location)]]
type_t shader_out;

template<typename type_t, int binding>
[[using spirv: buffer, binding(binding)]]
type_t shader_buffer;

template<typename type_t, int binding>
[[using spirv: uniform, binding(binding)]]
type_t shader_uniform;

struct push_t {
  MatrixData* matrixData;
  MaterialData* materialData;
};
[[spirv::push]]
push_t push;

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

[[spirv::vert]]
void vert_shader() {
  vec4 inPosNormal = shader_in<vec4, VERTEX_POS_OCTNORMAL>;

  vec3 inNormal = oct_to_float32x3(unpackSnorm2x16(floatBitsToUint(inPosNormal.w)));

  mat4 worldMatrix   = push.matrixData->worldMatrix;
  mat4 worldMatrixIT = push.matrixData->worldMatrixIT;

  vec3 wPos     = (worldMatrix   * vec4(inPosNormal.xyz, 1)).xyz;
  vec3 wNormal  = mat3(worldMatrixIT) * inNormal;

  mat4 viewProjMatrix = shader_uniform<SceneData, 0>.viewProjMatrix;
  glvert_Output.Position = viewProjMatrix * vec4(wPos, 1);

  shader_out<interpolants_t, 0> = {
    wPos, wNormal, inNormal, 
  };
}

template<int permutation>
[[spirv::frag]]
void frag_shader() {
  MaterialSide side = push.materialData->sides[glfrag_FrontFacing];

  // Apply shading.
  vec4 color = side.ambient + side.emissive;

  // Apply stiple. 
  ivec2 pixel = ivec2(glfrag_FragCoord.xy);
  pixel /= (permutation % 8) + 1;
  pixel %= (permutation % 2) + 1;
  pixel = 1 - pixel;

  interpolants_t in = shader_in<interpolants_t, 0>;

  color = mix(color, vec4(.5f * in.oNormal + .5f, 1), .5f * pixel.x * pixel.y);
  color += .001f * permutation;

  const SceneData& scene = shader_uniform<SceneData, 0>;
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
  float sequence = movement*2+3;  // time for sequence
  
  float timeS = fract(anim.time / sequence) * sequence;
  float time  = 
    clamp(timeS -      s  * movement,                   0.f, 1.f) - 
    clamp(timeS - (1 - s) * movement - sequence * 0.5f, 0.f, 1.f);
  
  float scale = smoothstep(0,1,time);
  
  mat4 matrixOrig = original[gid].worldMatrix;
  vec3 pos  = matrixOrig[3].xyz;
  vec3 away = (pos - anim.sceneCenter);
  
  float diridx = gid % 3;
  float sidx   = gid % 6;

  vec3 delta(
    diridx == 0,
    diridx == 1,
    diridx == 2
  );

  delta *= -sign(sidx - 2.5f);
  delta *= sign(dot(away,delta));
  
  delta = normalize(delta);
  pos += delta * scale * anim.sceneDimension;
  
  animated[gid].worldMatrix = mat4(
    matrixOrig[0],
    matrixOrig[1], 
    matrixOrig[2], 
    vec4(pos, 1)
  );
}

struct shaders_t {
  const char* spirv_data;
  size_t spirv_size;

  const char* comp_animation;
  const char* vert;
  const char* frags[128];
};

const shaders_t shaders {
  __spirv_data,
  __spirv_size,

  @spirv(comp_animation),
  @spirv(vert_shader),
  @spirv(frag_shader<__integer_pack(128)>)...
};