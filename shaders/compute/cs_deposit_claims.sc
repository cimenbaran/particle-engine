#include <bgfx_compute.sh>
BUFFER_RO(b_source,vec4,0);
BUFFER_RO(b_candidates,uvec4,1);
BUFFER_RO(b_preflight,uint,2);
BUFFER_RW(b_owners,uint,3);
BUFFER_RW(b_claims,uvec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_deposit_claims; // capacity, quota
bool depositBlocks(vec4 p,ivec3 cell){
 vec3 low=vec3(cell)*0.125,high=low+vec3_splat(0.125);
 // Exact dyadic separating planes preserve common-radius face tangency.
 if(p.w==0.0625 && (any(lessThanEqual(p.xyz,low-vec3_splat(0.0625))) || any(greaterThanEqual(p.xyz,high+vec3_splat(0.0625)))))return false;
 vec3 delta=max(max(low-p.xyz,p.xyz-high),vec3_splat(0.0));
 // Ambiguous edge/corner distances defer rather than allowing penetration.
 return dot(delta,delta)<=p.w*p.w*1.00001+1e-10;
}
NUM_THREADS(1,1,1)
void main(){
 uint capacity=uint(u_deposit_claims.x),quota=uint(u_deposit_claims.y);
 for(uint slot=0u;slot<capacity;++slot)b_owners[slot]=0u;
 for(uint c=0u;c<quota;++c){b_claims[c*2u]=uvec4(0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu);b_claims[c*2u+1u]=uvec4(0xffffffffu,0u,0xffffffffu,0u);}
 b_status[0]=b_preflight[0];b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;
 if(b_preflight[0]!=0u)return;
 uint accepted=0u;
 LOOP for(uint first=0u;first<capacity && accepted<quota;++first){
  uvec4 target=b_candidates[first*2u],owner=b_candidates[first*2u+1u];
  if(owner.z!=2u || owner.w==0u || b_owners[first]!=0u)continue;
  bool claimed=false;LOOP for(uint c=0u;c<accepted;++c){uvec4 previous=b_claims[c*2u];if(previous.x==target.x && previous.y==target.y && previous.z==target.z){claimed=true;break;}}
  if(claimed)continue;
  vec3 centre=(vec3(ivec3(target.xyz))+vec3_splat(0.5))*0.125;
  uint second=0xffffffffu;float nearest=1.0;
  LOOP for(uint slot=0u;slot<capacity;++slot){
   if(slot==first || b_owners[slot]!=0u)continue;
   uvec4 other=b_candidates[slot*2u+1u];if(other.z!=2u || b_candidates[slot*2u].w!=target.w)continue;
   vec3 position=b_source[slot*4u].xyz;
   if(any(lessThan(position,centre-vec3_splat(0.1875))) || any(greaterThan(position,centre+vec3_splat(0.1875))))continue;
   vec3 delta=position-centre;float squared=dot(delta,delta);
   if(squared<nearest){nearest=squared;second=slot;}
  }
  if(second==0xffffffffu)continue;
  bool blocked=false;
  LOOP for(uint slot=0u;slot<capacity;++slot){
   if(slot==first || slot==second || b_owners[slot]!=0u || b_candidates[slot*2u+1u].z==0u)continue;
   if(depositBlocks(b_source[slot*4u],ivec3(target.xyz))){blocked=true;break;}
  }
  if(blocked){++b_status[3];continue;}
  b_claims[accepted*2u]=target;
  b_claims[accepted*2u+1u]=uvec4(first,owner.y,second,b_candidates[second*2u+1u].y);
  ++accepted;b_owners[first]=accepted;b_owners[second]=accepted;
 }
 b_status[1]=accepted;b_status[2]=accepted*2u;
}
