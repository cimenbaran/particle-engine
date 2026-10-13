#include "granular_grid.sh"
#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_records,uint,0);
BUFFER_RW(b_flags,uint,1);
BUFFER_RW(b_meta,uint,2);
BUFFER_RW(b_owners,uint,3);
uniform vec4 u_ranges;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=uint(u_ranges.x)) return;
  b_flags[id]=0u;
  uint key=b_records[id*2u],slot=b_records[id*2u+1u];
  if(slot>=uint(u_ranges.x)) atomicOr(b_meta[2],32u);
  else atomicAdd(b_owners[slot],1u);
  if(key==0xffffffffu) return;
  atomicAdd(b_meta[1],1u);
  uint previous=b_records[(id>0u?id-1u:0u)*2u];
  if(id>0u && previous>key) atomicOr(b_meta[2],4u);
  if(key>=PW_GRID_KEYS) {atomicOr(b_meta[2],8u);return;}
  if(id==0u || previous!=key) b_flags[id]=1u;
}
