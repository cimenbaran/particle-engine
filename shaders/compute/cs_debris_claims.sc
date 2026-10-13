#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_snapshot,vec4,0);
BUFFER_RO(b_claims,uvec4,1);
BUFFER_RW(b_owners,uint,2);
BUFFER_RW(b_status,uint,3);
uniform vec4 u_debris_activation;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint rank=gl_GlobalInvocationID.x;
  if(rank>=uint(u_debris_activation.y) || (b_status[0]&1u)!=0u)return;
  uvec4 claim=b_claims[rank];
  if(claim.x>=uint(u_debris_activation.x) || claim.y==0u || claim.z==0xffffffffu || claim.y!=claim.z+1u || claim.w!=0u) {atomicOr(b_status[0],2u);return;}
  if(rank>0u && b_claims[rank-1u].x>=claim.x){atomicOr(b_status[0],2u);return;}
  uvec4 identity=floatBitsToUint(b_snapshot[claim.x*4u+3u]);
  if(identity.y!=2u || identity.z!=claim.x || identity.w!=claim.y){atomicOr(b_status[0],2u);return;}
  uint previous;atomicFetchCompareExchange(b_owners[claim.x],0xffffffffu,rank,previous);
  if(previous!=0xffffffffu)atomicOr(b_status[0],2u);
}
