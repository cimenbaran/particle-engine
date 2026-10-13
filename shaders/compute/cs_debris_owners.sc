#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_snapshot,vec4,0);
BUFFER_RW(b_owners,uint,1);
BUFFER_RW(b_status,uint,2);
uniform vec4 u_debris_activation;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_debris_activation.x))return;
  b_owners[slot]=0xffffffffu;
  uvec4 identity=floatBitsToUint(b_snapshot[slot*4u+3u]);
  if(identity.y==2u)atomicAdd(b_status[2],1u);
  if(identity.y>2u || (identity.y!=0u && (identity.z!=slot || identity.w==0u)))atomicOr(b_status[0],2u);
}
