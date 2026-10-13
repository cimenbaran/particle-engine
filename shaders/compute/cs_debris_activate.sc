#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_snapshot,vec4,0);
BUFFER_RO(b_seeds,vec4,1);
BUFFER_RO(b_owners,uint,2);
BUFFER_RO(b_status,uint,3);
BUFFER_RW(b_output,vec4,4);
uniform vec4 u_debris_activation;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_debris_activation.x))return;
  uint base=slot*4u,rank=b_owners[slot];
  if(b_status[0]!=0u || rank==0xffffffffu) {
    for(uint lane=0u;lane<4u;++lane)b_output[base+lane]=b_snapshot[base+lane];return;
  }
  for(uint lane=0u;lane<3u;++lane)b_output[base+lane]=b_seeds[rank*4u+lane];
  uvec4 identity=floatBitsToUint(b_snapshot[base+3u]);
  identity.x=floatBitsToUint(b_seeds[rank*4u+3u]).x;identity.y=1u;
  b_output[base+3u]=uintBitsToFloat(identity);
}
