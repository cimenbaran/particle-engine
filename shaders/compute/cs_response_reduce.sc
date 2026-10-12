#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_result,vec4,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_response;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_response.x)) return;
  uvec4 info=floatBitsToUint(b_result[id*2u+1u]);atomicOr(b_status[0],info.x);
  atomicMax(b_status[1],info.y);atomicMax(b_status[2],info.z);atomicMax(b_status[3],info.w);
}
