#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_output,uint,0);
BUFFER_RO(b_offsets,uint,1);
uniform vec4 u_scan;
uint scanSum(uint a,uint b) {
  uint sum=a+b;
#ifdef PW_SCAN_SATURATE
  if(sum<a) return 0xffffffffu;
#endif
  return sum;
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id<uint(u_scan.x)) b_output[id]=scanSum(b_output[id],b_offsets[gl_WorkGroupID.x]);
}
