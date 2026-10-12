#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_owners,uint,0);
BUFFER_RW(b_meta,uint,1);
uniform vec4 u_ranges;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id<uint(u_ranges.x) && b_owners[id]!=1u) atomicOr(b_meta[2],64u);
}
