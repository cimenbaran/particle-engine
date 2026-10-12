#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_table,uint,1);
BUFFER_RW(b_meta,uint,2);
BUFFER_RW(b_metrics,uint,3);
uniform vec4 u_hash;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id<uint(u_hash.y)) {b_table[id*2u]=0xffffffffu;b_table[id*2u+1u]=0xffffffffu;}
  if(id!=0u) return;
  uint count=b_source[0],records=b_source[1],capacity=uint(u_hash.x),errors=0u;
  if(b_source[2]!=0u) errors|=1u;
  if(count>capacity || records>capacity || count>records || (count==0u)!=(records==0u) ||
     (b_source[2]==0u && count!=b_source[3])) errors|=2u;
  // Division avoids overflow even for corrupt source counts.
  if(count>=uint(u_hash.y)/2u) errors|=2u;
  b_meta[0]=0u;b_meta[1]=0u;b_meta[2]=0u;b_meta[3]=errors;
  b_metrics[0]=0u;b_metrics[1]=0u;b_metrics[2]=0u;b_metrics[3]=0u;
}
