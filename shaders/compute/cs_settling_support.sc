#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_support,vec4,5);
uniform vec4 u_neighbors;
#include "contact_math.sh"
void addSupport(uint query,float gap) {
  if(gap< -0.00625)return;
  vec4 s=b_support[query];s.x=max(s.x,gap<=0.0001?1.0:0.0);s.y=max(s.y,gap<=0.002?1.0:0.0);b_support[query]=s;
}
void supportContact(uint queryIndex,uint slot,vec3 delta,float distanceSquared) {
  if(distanceSquared<=1e-12)return;
  float distance=sqrt(distanceSquared);
  if(-delta.y>=0.5*distance)addSupport(b_sorted[queryIndex*2u+1u],distance-0.125);
}
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) supportContact(_query,_slot,_delta,_distance)
#include "neighbors.sh"
bool finiteValue(float v){return (floatBitsToUint(v)&0x7f800000u)!=0x7f800000u;}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_neighbors.x))return;
  if(b_meta[0]!=1u){b_support[id]=vec4(0.0,0.0,uintBitsToFloat(2u),0.0);return;}
  uint slot=b_sorted[id*2u+1u];b_support[slot]=vec4_splat(0.0);
  if(b_sorted[id*2u]==0xffffffffu)return;
  vec4 velocity=b_state[slot*4u+1u];
  if(b_state[slot*4u].w!=0.0625 || !validMass(velocity.w) ||
     !finiteValue(velocity.x) || !finiteValue(velocity.y) || !finiteValue(velocity.z)) {
    b_support[slot]=vec4(0.0,0.0,uintBitsToFloat(4u),0.0);return;
  }
  if(velocity.w==0.0)return;
  b_support[slot]=vec4(0.0,0.0,0.0,1.0);addSupport(slot,b_state[slot*4u].y-0.0625);
  uvec4 result=visitNeighbors(id,u_neighbors);
  if(result.w>1u){vec4 s=b_support[slot];s.z=uintBitsToFloat(8u);b_support[slot]=s;}
}
