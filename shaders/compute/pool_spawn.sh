#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "materials.sh"
BUFFER_RW(b_state, vec4, 0);
BUFFER_RW(b_counts, uint, 1);
BUFFER_RO(b_free, uint, 2);
#if PW_VOXEL_SPAWN
BUFFER_RO(b_world,uvec4,4);
uniform vec4 u_spawn_lattice_min;
uniform vec4 u_spawn_lattice_size;
#include "contact_math.sh"
#include "voxel_world_contact.sh"
#endif
uniform vec4 u_pool; // capacity, requested count, material count, seed bits
uniform vec4 u_brushMin; // xyz, material ID
uniform vec4 u_brushMax; // xyz, radius
uniform vec4 u_brushValue; // velocity xyz, lifetime seconds (0=infinite)
uint hash32(uint x) {x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
float random01(uint x) {return float(hash32(x)>>8u)/16777216.0;}
void rejectSpawn() {
  uint observed; atomicFetchAndAdd(b_counts[3],0u,observed);
  while(observed!=0xffffffffu) {
    uint previous; atomicFetchCompareExchange(b_counts[3],observed,observed+1u,previous);
    if(previous==observed) return;
    observed=previous;
  }
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint request=gl_GlobalInvocationID.x;
  if(request>=uint(u_pool.y)) return;
  uint material=uint(u_brushMin.w);
  uint seed=floatBitsToUint(u_pool.w)^hash32(request);
  vec3 random=vec3(random01(seed),random01(seed+1u),random01(seed+2u));
  vec3 position=mix(u_brushMin.xyz,u_brushMax.xyz,random);
#if PW_VOXEL_SPAWN
  if(u_spawn_lattice_min.w!=0.0) {
    uint sites=floatBitsToUint(u_spawn_lattice_size.w);
    if(request>=sites){rejectSpawn();return;}
    uvec3 dimensions=uvec3(u_spawn_lattice_size.xyz);
    uint index=(hash32(floatBitsToUint(u_pool.w))%sites+request)%sites;
    uvec3 cell=uvec3(index%dimensions.x,(index/dimensions.x)%dimensions.y,index/(dimensions.x*dimensions.y));
    position=(u_spawn_lattice_min.xyz+vec3(cell))*.125;
  }
  vec3 centre,normal;float time;uint terrainMaterial,index;
  if(material>=uint(u_pool.z) || floatBitsToUint(b_materials[material*4u+3u]).y!=0u ||
     voxelWorldSweep(position,position,false,0xffffffffu,centre,normal,time,terrainMaterial,index)>1u) {
    rejectSpawn();return;
  }
#endif
  uint rank;
  atomicFetchAndAdd(b_counts[2],1u,rank);
  if(rank>=b_counts[1] || material>=uint(u_pool.z)) {rejectSpawn();return;}
  uint slot=b_free[rank];
  if(slot>=uint(u_pool.x)) {rejectSpawn();return;}
  uint base=slot*4u;
  uvec4 previous=floatBitsToUint(b_state[base+3u]);
  // Never recycle a wrapped generation into a stale valid handle.
  if(previous.w==0xffffffffu) {rejectSpawn();return;}
  float radius=u_brushMax.w;
  float mass=(4.0/3.0)*3.14159265*radius*radius*radius*b_materials[material*4u].x;
  b_state[base]=vec4(position,radius);
  b_state[base+1u]=vec4(u_brushValue.xyz,1.0/mass);
  b_state[base+2u]=vec4(293.15,0.0,u_brushValue.w,0.0);
  b_state[base+3u]=uintBitsToFloat(uvec4(material,1u,slot,previous.w+1u));
}
