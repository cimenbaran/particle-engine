#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_records,vec4,0);
BUFFER_RW(b_totals,uint,1);
BUFFER_RO(b_phase,uint,2);
uniform vec4 u_contact_totals; // capacity, 0 Jacobi / 1 response
SHARED uvec4 partial[PW_WORKGROUP_SIZE];
uvec2 sum64(uvec2 a,uvec2 b){uint low=a.x+b.x;return uvec2(low,a.y+b.y+(low<a.x?1u:0u));}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint lane=gl_LocalInvocationID.x,id=gl_GlobalInvocationID.x;uvec4 value=uvec4(0u,0u,0u,0u);
  if(id<uint(u_contact_totals.x) && b_phase[0]==0u) {
    uvec4 row=floatBitsToUint(b_records[id*2u+1u]);
    value=u_contact_totals.y==0.0?uvec4(row.y,0u,row.x,0u):uvec4(sum64(uvec2(row.z,0u),uvec2(row.z,0u)),row.y,0u);
  }
  partial[lane]=value;barrier();
  for(uint stride=PW_WORKGROUP_SIZE/2u;stride>0u;stride/=2u){
    if(lane<stride){uvec4 a=partial[lane],b=partial[lane+stride];partial[lane]=uvec4(sum64(a.xy,b.xy),sum64(a.zw,b.zw));}
    barrier();
  }
  if(lane==0u){
    uvec4 total=partial[0];uint oldCandidates,oldContacts;
    atomicFetchAndAdd(b_totals[0],total.x,oldCandidates);
    atomicAdd(b_totals[1],total.y+(oldCandidates+total.x<oldCandidates?1u:0u));
    atomicFetchAndAdd(b_totals[2],total.z,oldContacts);
    atomicAdd(b_totals[3],total.w+(oldContacts+total.z<oldContacts?1u:0u));
  }
}
