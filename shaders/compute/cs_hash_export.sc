#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_results,uint,0);
UIMAGE2D_WO(o_result,rgba32ui,1);
uniform vec4 u_hash;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_hash.w)) return;
  imageStore(o_result,ivec2(int(id%256u),int(id/256u)),
    uvec4(b_results[id*4u],b_results[id*4u+1u],b_results[id*4u+2u],b_results[id*4u+3u]));
}
