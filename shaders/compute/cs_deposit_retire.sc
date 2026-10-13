#include <bgfx_compute.sh>
BUFFER_RO(b_source,uvec4,0);
BUFFER_RO(b_claims,uvec4,1);
BUFFER_RO(b_owners,uint,2);
BUFFER_RO(b_claimStatus,uint,3);
BUFFER_RW(b_output,uvec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_deposit_claims;
// Bounded serial validation completes before any output mutation. Exact integer
// lanes preserve all payload bits, including generations above float precision.
NUM_THREADS(1,1,1)
void main(){
 uint capacity=uint(u_deposit_claims.x),quota=uint(u_deposit_claims.y);
 uint errors=b_claimStatus[0],count=b_claimStatus[1],owned=0u;
 if(count>quota || count>capacity/2u || b_claimStatus[2]!=count*2u)errors|=64u;
 if(errors==0u){
  for(uint c=0u;c<count;++c){
   uvec4 target=b_claims[c*2u],ids=b_claims[c*2u+1u];
   if(ids.x>=capacity || ids.z>=capacity || ids.x==ids.z){errors|=64u;continue;}
   uvec4 a=b_source[ids.x*4u+3u],b=b_source[ids.z*4u+3u];
   if(a.x!=target.w || b.x!=target.w || a.y!=1u || b.y!=1u || a.z!=ids.x || b.z!=ids.z || a.w==0u || b.w==0u || a.w!=ids.y || b.w!=ids.w)errors|=128u;
   if(b_owners[ids.x]!=c+1u || b_owners[ids.z]!=c+1u)errors|=64u;
  }
  for(uint slot=0u;slot<capacity;++slot){
   uint owner=b_owners[slot];if(owner==0u)continue;++owned;
   if(owner>count){errors|=64u;continue;}
   uvec4 ids=b_claims[(owner-1u)*2u+1u];
   if(slot!=ids.x && slot!=ids.z)errors|=64u;
  }
  if(owned!=count*2u)errors|=64u;
 }
 for(uint slot=0u;slot<capacity;++slot){
  for(uint lane=0u;lane<4u;++lane)b_output[slot*4u+lane]=b_source[slot*4u+lane];
  if(errors==0u && b_owners[slot]!=0u){uvec4 id=b_source[slot*4u+3u];id.y=0u;b_output[slot*4u+3u]=id;}
 }
 b_status[0]=errors;b_status[1]=errors==0u?count:0u;b_status[2]=errors==0u?owned:0u;b_status[3]=0u;
}
