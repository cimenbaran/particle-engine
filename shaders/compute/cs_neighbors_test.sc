#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_results,uint,5);
uniform vec4 u_neighbors;
void recordNeighbor(uint queryIndex,uint slot) {
  uint stride=(uint(u_neighbors.x)+31u)/32u+4u;
  uint offset=queryIndex*stride+slot/32u;
  b_results[offset]=b_results[offset]|(1u<<(slot%32u));
}
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) recordNeighbor(_query,_slot)
#include "neighbors.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x,capacity=uint(u_neighbors.x);if(id>=capacity) return;
  uint words=(capacity+31u)/32u,stride=words+4u;
  for(uint i=0u;i<words;++i) b_results[id*stride+i]=0u;
  uvec4 result=visitNeighbors(id,u_neighbors);
  b_results[id*stride+words]=result.x;b_results[id*stride+words+1u]=result.y;
  b_results[id*stride+words+2u]=result.z;b_results[id*stride+words+3u]=result.w;
}
