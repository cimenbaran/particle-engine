#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_results,uint,0);
UIMAGE2D_WO(o_result,rgba32ui,1);
uniform vec4 u_neighbors;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint count=uint(u_neighbors.x)*((uint(u_neighbors.x)+31u)/32u+4u);
  uint id=gl_GlobalInvocationID.x;if(id>=(count+3u)/4u) return;
  uvec4 value=uvec4(0u,0u,0u,0u);
  for(uint i=0u;i<4u;++i) if(id*4u+i<count) value[int(i)]=b_results[id*4u+i];
  imageStore(o_result,ivec2(int(id%256u),int(id/256u)),value);
}
