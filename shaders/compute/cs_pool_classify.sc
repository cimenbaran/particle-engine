#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state, vec4, 0);
BUFFER_RW(b_counts, uint, 1);
BUFFER_RW(b_free, uint, 2);
BUFFER_RW(b_active, uint, 3);
uniform vec4 u_pool;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;
  if(slot>=uint(u_pool.x)) return;
  uvec4 identity=floatBitsToUint(b_state[slot*4u+3u]);
  uint rank;
  if((identity.y&1u)!=0u) {
    atomicFetchAndAdd(b_counts[0],1u,rank); b_active[rank]=slot;
  } else {
    atomicFetchAndAdd(b_counts[1],1u,rank); b_free[rank]=slot;
  }
}
