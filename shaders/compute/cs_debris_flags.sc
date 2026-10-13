#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,vec4,0);
BUFFER_RW(b_flags,uint,1);
BUFFER_RW(b_status,uint,2);
BUFFER_RW(b_claims,uvec4,3);
uniform vec4 u_debris;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_debris.x))return;
  uvec4 identity=floatBitsToUint(b_source[slot*4u+3u]);
  bool valid=identity.y<=1u && (identity.y==0u || (identity.z==slot && identity.w!=0u));
  if(!valid)atomicOr(b_status[0],2u);
  b_flags[slot]=valid && identity.y==0u && identity.w!=0xffffffffu?1u:0u;
  b_claims[slot]=uvec4(0xffffffffu,0u,0u,0u);
}
