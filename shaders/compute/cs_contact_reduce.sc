#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_corrections,vec4,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_contacts;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_contacts.x)) return;
  uvec4 stats=floatBitsToUint(b_corrections[id*2u+1u]);
  if(stats.w>1u) atomicOr(b_status[0],1u<<(stats.w-2u));
  atomicMax(b_status[1],stats.x);atomicMax(b_status[2],stats.y);atomicMax(b_status[3],stats.z);
}
