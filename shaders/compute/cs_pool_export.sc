#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_counts, uint, 0);
BUFFER_RO(b_free, uint, 1);
BUFFER_RO(b_active, uint, 2);
IMAGE2D_RW(o_data, rgba32ui, 3);
uniform vec4 u_pool;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>uint(u_pool.x)) return;
  uvec4 value=uvec4(0xffffffffu);
  if(id==uint(u_pool.x)) value=uvec4(b_counts[0],b_counts[1],b_counts[2],b_counts[3]);
  else {
    if(id<b_counts[1]) value.x=b_free[id];
    if(id<b_counts[0]) value.y=b_active[id];
  }
  imageStore(o_data,ivec2(int(id%256u),int(id/256u)),value);
}
