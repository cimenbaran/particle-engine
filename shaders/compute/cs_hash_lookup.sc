#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_ranges,uint,0);
BUFFER_RO(b_table,uint,1);
BUFFER_RO(b_meta,uint,2);
BUFFER_RO(b_keys,uint,3);
BUFFER_RW(b_results,uint,4);
#include "cell_lookup.sh"
uniform vec4 u_hash;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_hash.w)) return;
  uint begin=0u,end=0u,status=2u,probes=0u;
  if(b_meta[0]==1u) status=lookupCell(b_keys[id],uint(u_hash.x),uint(u_hash.y),uint(u_hash.z),begin,end,probes);
  b_results[id*4u]=begin;b_results[id*4u+1u]=end;
  b_results[id*4u+2u]=status;b_results[id*4u+3u]=probes;
}
