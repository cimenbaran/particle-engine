#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_active,uint,1);
BUFFER_RO(b_free,uint,2);
BUFFER_RO(b_pool,uint,3);
BUFFER_RW(b_owners,uint,4);
BUFFER_RW(b_result,uint,5);
uniform vec4 u_audit;
void checkOwner(uint slot,bool alive) {
  if(slot>=uint(u_audit.x)) {atomicOr(b_result[1],1u);return;}
  atomicAdd(b_owners[slot],1u);
  bool actual=(floatBitsToUint(b_state[slot*4u+3u]).y&1u)!=0u;
  if(actual!=alive) atomicOr(b_result[1],1u);
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x,capacity=uint(u_audit.x);
  if(id>=capacity) return;
  if(id==0u && (b_pool[0]>capacity || b_pool[1]>capacity || b_pool[0]+b_pool[1]!=capacity)) atomicOr(b_result[1],1u);
  if(id<min(b_pool[0],capacity)) checkOwner(b_active[id],true);
  if(id<min(b_pool[1],capacity)) checkOwner(b_free[id],false);
}
