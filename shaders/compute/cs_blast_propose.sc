#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_source,vec4,0);
BUFFER_RO(b_materials,vec4,1);
BUFFER_RW(b_candidate,vec4,2);
BUFFER_RW(b_costs,uint,3);
BUFFER_RW(b_status,uint,4);
uniform vec4 u_blast; // capacity, material count, ticket budget, joules/ticket
uniform vec4 u_blast_centre; // xyz, radius
uniform vec4 u_blast_speed;
bool finiteBlast(float v){return (floatBitsToUint(v)&0x7f800000u)!=0x7f800000u;}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_blast.x))return;
  uint base=slot*4u;for(uint lane=0u;lane<4u;++lane)b_candidate[base+lane]=b_source[base+lane];
  b_costs[slot]=0u;
  uvec4 id=floatBitsToUint(b_source[base+3u]);
  if(id.y==0u)return;
  if(id.y!=1u || id.z!=slot || id.w==0u){atomicOr(b_status[0],1u);return;}
  if(id.x>=uint(u_blast.y)){atomicOr(b_status[0],4u);return;}
  uvec4 material=floatBitsToUint(b_materials[id.x*4u+3u]);
  if(material.x!=id.x || material.y>2u){atomicOr(b_status[0],4u);return;}
  if(material.y!=0u)return;
  vec4 position=b_source[base],velocity=b_source[base+1u];
  if(!finiteBlast(position.x) || !finiteBlast(position.y) || !finiteBlast(position.z) || !finiteBlast(position.w) ||
     !finiteBlast(velocity.x) || !finiteBlast(velocity.y) || !finiteBlast(velocity.z) || !finiteBlast(velocity.w) ||
     any(greaterThan(abs(position.xyz),vec3(1000000.0))) || position.w<=0.0 || position.w>1.0 || velocity.w<0.0 ||
     (velocity.w>0.0 && (velocity.w<0.000001 || velocity.w>1000000.0)) || dot(velocity.xyz,velocity.xyz)>144.001)
     {atomicOr(b_status[0],2u);return;}
  if(velocity.w==0.0 || u_blast_speed.x==0.0)return;
  vec3 offset=position.xyz-u_blast_centre.xyz;float distance=length(offset);
  if(distance>=u_blast_centre.w)return;
  vec3 direction=distance<0.000001?vec3(0.0,1.0,0.0):offset/distance;
  float falloff=max(0.0,1.0-distance/u_blast_centre.w);
  vec3 changed=velocity.xyz+direction*(u_blast_speed.x*falloff*falloff);
  float changedSpeed=length(changed);
  if(changedSpeed>11.99998)changed*=11.99998/changedSpeed;
  if(all(equal(changed,velocity.xyz)))return;
  float delta=length(changed-velocity.xyz),oldSpeed=length(velocity.xyz);
  // Positive-work bound, including pre-existing motion. Inflation protects
  // binary32 length/division/product rounding in the validated numeric domain.
  float bound=((oldSpeed+0.5*delta)*delta/velocity.w)*1.00001+0.000000000001;
  float ratio=bound/u_blast.w;
  uint budget=uint(u_blast.z),cost=budget+1u;
  if(ratio<float(budget))cost=uint(ceil(ratio))+1u;
  b_costs[slot]=cost;b_candidate[base+1u]=vec4(changed,velocity.w);
  atomicAdd(b_status[1],1u);
}
