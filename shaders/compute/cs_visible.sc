#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_input, vec4, 0);
BUFFER_RW(b_output, vec4, 1);
BUFFER_RW(b_counts, uint, 2);
BUFFER_RO(b_active, uint, 3);
BUFFER_RO(b_pool, uint, 4);
uniform vec4 u_cull; // capacity, enable frustum culling
uniform vec4 u_frustum[6];
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint index=gl_GlobalInvocationID.x;
  if(index>=min(b_pool[0],uint(u_cull.x))) return;
  uint slot=b_active[index];
  if(slot>=uint(u_cull.x)) return;
  vec4 sphere=b_input[slot*2u];
  uvec4 visual=floatBitsToUint(b_input[slot*2u+1u]);
  if((visual.w&1u)==0u) return;
  if(u_cull.y>0.0) for(uint plane=0u;plane<6u;++plane)
    if(dot(u_frustum[plane].xyz,sphere.xyz)+u_frustum[plane].w < -sphere.w) return;
  uint rank;atomicFetchAndAdd(b_counts[0],1u,rank);
  if(rank>=uint(u_cull.x)) return;
  b_output[rank*2u]=sphere;b_output[rank*2u+1u]=b_input[slot*2u+1u];
}
