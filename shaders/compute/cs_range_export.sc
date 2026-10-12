#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_ranges,uint,0);
BUFFER_RO(b_meta,uint,1);
UIMAGE2D_WO(o_result,rgba32ui,2);
uniform vec4 u_ranges;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=max(1u,uint(u_ranges.x))) return;
  uvec4 value=uvec4(0xffffffffu,0xffffffffu,0xffffffffu,id);
  if(id<b_meta[3]) value=uvec4(b_ranges[id*3u],b_ranges[id*3u+1u],b_ranges[id*3u+2u],id);
  imageStore(o_result,ivec2(int(id%256u),int(id/256u)),value);
}
