#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_normals,vec4,5);
uniform vec4 u_neighbors;
uniform vec4 u_normal_sphere;
void emptyNormal(uint slot,uint error) {
  b_normals[slot*2u]=vec4(0.0,0.0,0.0,3.402823466e38);
  b_normals[slot*2u+1u]=uintBitsToFloat(uvec4(0u,0u,error,0u));
}
void considerNormal(uint slot,float gap,vec3 normal,uint source,uint feature) {
  vec4 old=b_normals[slot*2u];uvec4 meta=floatBitsToUint(b_normals[slot*2u+1u]);
  if(meta.z!=0u)return;
  bool priority=source<meta.x || (source==meta.x && feature<meta.y);
  if(gap<=0.002 && (gap<old.w || (gap==old.w && priority))) {
    b_normals[slot*2u]=vec4(normal,gap);
    b_normals[slot*2u+1u]=uintBitsToFloat(uvec4(source,feature,0u,0u));
  }
}
void normalContact(uint query,uint slot,vec3 delta,float squared) {
  uint id=b_sorted[query*2u+1u];
  if(b_state[slot*4u].w!=0.0625 || !validMass(b_state[slot*4u+1u].w)){emptyNormal(id,2u);return;}
  considerNormal(id,sqrt(squared)-0.125,contactNormal(id,slot,delta,squared),3u,slot);
}
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) normalContact(_query,_slot,_delta,_distance)
#include "neighbors.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_neighbors.x))return;
  if(b_meta[0]!=1u){emptyNormal(id,1u);return;}
  uint slot=b_sorted[id*2u+1u];emptyNormal(slot,0u);
  if(b_sorted[id*2u]==0xffffffffu)return;
  vec4 p=b_state[slot*4u],v=b_state[slot*4u+1u];
  if(p.w!=0.0625 || !validMass(v.w)){emptyNormal(slot,2u);return;}
  if(v.w==0.0)return;
  considerNormal(slot,p.x+7.9375,vec3(1.0,0.0,0.0),1u,0u);
  considerNormal(slot,7.9375-p.x,vec3(-1.0,0.0,0.0),1u,1u);
  considerNormal(slot,p.y-0.0625,vec3(0.0,1.0,0.0),1u,2u);
  considerNormal(slot,15.9375-p.y,vec3(0.0,-1.0,0.0),1u,3u);
  considerNormal(slot,p.z+7.9375,vec3(0.0,0.0,1.0),1u,4u);
  considerNormal(slot,7.9375-p.z,vec3(0.0,0.0,-1.0),1u,5u);
  if(u_normal_sphere.w>0.0) {
    vec3 offset=p.xyz-u_normal_sphere.xyz;float distance=length(offset);
    if(distance==0.0){emptyNormal(slot,2u);return;}
    considerNormal(slot,distance-u_normal_sphere.w-0.0625,offset/distance,2u,0u);
  }
  uvec4 visited=visitNeighbors(id,u_neighbors);
  if(visited.w>1u)emptyNormal(slot,4u);
}
