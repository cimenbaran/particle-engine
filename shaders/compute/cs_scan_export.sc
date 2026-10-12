#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_output,uint,0);
BUFFER_RO(b_total,uint,1);
BUFFER_RO(b_input,uint,2);
UIMAGE2D_WO(o_result,rgba32ui,3);
uniform vec4 u_scan;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=max(1u,uint(u_scan.x))) return;
  uint outputValue=0u,inputValue=0u;
  if(id<uint(u_scan.x)) {outputValue=b_output[id];inputValue=b_input[id];}
  imageStore(o_result,ivec2(int(id%256u),int(id/256u)),uvec4(outputValue,b_total[0],inputValue,id));
}
