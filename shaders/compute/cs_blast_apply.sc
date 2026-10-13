#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,vec4,0);
BUFFER_RO(b_candidate,vec4,1);
BUFFER_RO(b_costs,uint,2);
BUFFER_RO(b_prefix,uint,3);
BUFFER_RW(b_output,vec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_blast;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_blast.x))return;
  uint cost=b_costs[slot],prefix=b_prefix[slot],budget=uint(u_blast.z);
  bool accept=b_status[0]==0u && cost>0u && prefix<=budget && cost<=budget-prefix;
  for(uint lane=0u;lane<4u;++lane)b_output[slot*4u+lane]=accept?b_candidate[slot*4u+lane]:b_source[slot*4u+lane];
  if(accept){atomicAdd(b_status[2],1u);atomicAdd(b_status[3],cost);}
}
