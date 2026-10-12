#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_meta,uint,1);
BUFFER_RW(b_owners,uint,2);
uniform vec4 u_ranges;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id<uint(u_ranges.x)) b_owners[id]=0u;
  if(id!=0u) return;
  uint errors=0u;
  if(b_source[2]!=0u) errors|=1u;
  if(b_source[3]!=0u) errors|=2u;
  uint count=uint(u_ranges.x);
  if(b_source[0]>count || b_source[1]>count || b_source[2]>count || b_source[3]>count ||
     b_source[0]+b_source[1]+b_source[2]+b_source[3]!=count) errors|=16u;
  b_meta[0]=0u;b_meta[1]=0u;b_meta[2]=errors;b_meta[3]=0u;
}
