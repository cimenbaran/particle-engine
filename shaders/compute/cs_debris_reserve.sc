#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,vec4,0);
BUFFER_RO(b_flags,uint,1);
BUFFER_RO(b_prefix,uint,2);
BUFFER_RO(b_status,uint,3);
BUFFER_RW(b_snapshot,vec4,4);
BUFFER_RW(b_claims,uvec4,5);
uniform vec4 u_debris;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_debris.x))return;
  uint base=slot*4u;
  for(uint lane=0u;lane<4u;++lane)b_snapshot[base+lane]=b_source[base+lane];
  uint rank=b_prefix[slot];
  if(b_status[0]!=0u || b_flags[slot]==0u || rank>=b_status[3])return;
  uvec4 identity=floatBitsToUint(b_source[base+3u]);uint previous=identity.w;
  identity.y=2u;identity.z=slot;identity.w=previous+1u;
  b_snapshot[base+3u]=uintBitsToFloat(identity);
  b_claims[rank]=uvec4(slot,identity.w,previous,0u);
}
