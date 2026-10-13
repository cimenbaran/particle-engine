#include "granular_grid.sh"
#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_records,uint,0);
BUFFER_RO(b_flags,uint,1);
BUFFER_RO(b_prefix,uint,2);
BUFFER_RW(b_ranges,uint,3);
uniform vec4 u_ranges;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=uint(u_ranges.x)) return;
  uint key=b_records[id*2u];
  if(key>=PW_GRID_KEYS) return;
  uint range=b_prefix[id]+b_flags[id]-1u;
  if(b_flags[id]!=0u) {b_ranges[range*3u]=key;b_ranges[range*3u+1u]=id;}
  if(id+1u==uint(u_ranges.x) || b_records[min(id+1u,uint(u_ranges.x)-1u)*2u]!=key) b_ranges[range*3u+2u]=id+1u;
}
