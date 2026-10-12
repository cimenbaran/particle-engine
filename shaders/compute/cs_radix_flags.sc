#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_records,uint,0);
BUFFER_RW(b_flags,uint,1);
uniform vec4 u_radix;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id<uint(u_radix.x)) b_flags[id]=1u-((b_records[id*2u]>>uint(u_radix.y))&1u);
}
