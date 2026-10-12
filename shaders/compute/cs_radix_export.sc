#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_sorted,uint,0);
BUFFER_RO(b_input,uint,1);
UIMAGE2D_WO(o_result,rgba32ui,2);
uniform vec4 u_radix;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=max(1u,uint(u_radix.x))) return;
  uvec4 result=uvec4(0u,0u,0u,0u);
  if(id<uint(u_radix.x)) result=uvec4(b_sorted[id*2u],b_sorted[id*2u+1u],b_input[id*2u],b_input[id*2u+1u]);
  imageStore(o_result,ivec2(int(id%256u),int(id/256u)),result);
}
