#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
uniform vec4 u_neighbors;
uniform vec4 u_response;
#include "response_pair.sh"
void countContact(uint queryIndex,uint slot,float squared) {
  uint query=b_sorted[queryIndex*2u+1u];uvec4 row=floatBitsToUint(b_state[query*8u+7u]);
  uint error=floatBitsToUint(b_state[slot*8u+6u].x);
  if(error!=0u) row.y|=error;
  else if(responsePair(query,slot,squared)) ++row.x;
  b_state[query*8u+7u]=uintBitsToFloat(row);
}
#define PW_STATE_STRIDE 8u
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) countContact(_query,_slot,_distance)
#include "neighbors.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_response.x)) return;
  if(b_meta[0]!=1u) {b_state[id*8u+7u]=uintBitsToFloat(uvec4(0u,2u,0u,0u));return;}
  uint slot=b_sorted[id*2u+1u],error=floatBitsToUint(b_state[slot*8u+6u].x);
  b_state[slot*8u+7u]=uintBitsToFloat(uvec4(0u,error,0u,0u));
  if(error!=0u) return;
  uvec4 result=visitNeighbors(id,u_neighbors),row=floatBitsToUint(b_state[slot*8u+7u]);
  if(result.w>1u) row.y|=2u;
  row.z=result.y;row.w=result.z;b_state[slot*8u+7u]=uintBitsToFloat(row);
}
