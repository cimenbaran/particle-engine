#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_corrections,vec4,5);
uniform vec4 u_neighbors;
uniform vec4 u_contacts; // capacity, common radius, relaxation, correction limit
#include "contact_math.sh"
bool validContactSlot(uint slot) {
  return b_state[slot*4u].w==u_contacts.y && validMass(b_state[slot*4u+1u].w);
}
void contact(uint queryIndex,uint slot,vec3 delta,float distanceSquared) {
  uint query=b_sorted[queryIndex*2u+1u];
  vec4 counts=b_corrections[query*2u+1u];
  if(!validContactSlot(slot)) {counts.w=uintBitsToFloat(4u);b_corrections[query*2u+1u]=counts;return;}
  float distance=sqrt(distanceSquared),penetration=2.0*u_contacts.y-distance;
  if(penetration<=0.0) return;
  counts.x=uintBitsToFloat(floatBitsToUint(counts.x)+1u);b_corrections[query*2u+1u]=counts;
  float a=b_state[query*4u+1u].w,b=b_state[slot*4u+1u].w;
  if(a==0.0) return;
  float weight=massWeight(a,b);
  vec3 direction=contactNormal(query,slot,delta,distanceSquared);
  vec4 correction=b_corrections[query*2u];correction.xyz+=direction*(penetration*weight);
  b_corrections[query*2u]=correction;
}
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) contact(_query,_slot,_delta,_distance)
#include "neighbors.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x,capacity=uint(u_contacts.x);if(id>=capacity) return;
  if(b_meta[0]!=1u) {
    b_corrections[id*2u]=vec4(0.0,0.0,0.0,0.0);
    b_corrections[id*2u+1u]=uintBitsToFloat(uvec4(0u,0u,0u,2u));return;
  }
  // A ready index proves the sorted payload is a permutation of physical slots.
  uint slot=b_sorted[id*2u+1u];
  b_corrections[slot*2u]=vec4(0.0,0.0,0.0,0.0);b_corrections[slot*2u+1u]=vec4(0.0,0.0,0.0,0.0);
  uvec4 result;
  if(b_sorted[id*2u]!=0xffffffffu && !validContactSlot(slot)) result=uvec4(0u,0u,0u,4u);
  else result=visitNeighbors(id,u_neighbors);
  uvec4 counts=floatBitsToUint(b_corrections[slot*2u+1u]);
  counts.y=result.y;counts.z=result.z;counts.w=max(counts.w,result.w);
  vec3 correction=b_corrections[slot*2u].xyz*u_contacts.z;
  float lengthSquared=dot(correction,correction);
  if(lengthSquared>u_contacts.w*u_contacts.w) correction*=u_contacts.w/sqrt(lengthSquared);
  if((floatBitsToUint(correction.x)&0x7f800000u)==0x7f800000u ||
     (floatBitsToUint(correction.y)&0x7f800000u)==0x7f800000u ||
     (floatBitsToUint(correction.z)&0x7f800000u)==0x7f800000u) counts.w=5u;
  vec3 proposed=b_state[slot*4u].xyz+correction;
  if(counts.w==0u && (proposed.x< -8.0 || proposed.x>=8.0 || proposed.y<0.0 || proposed.y>=16.0 || proposed.z< -8.0 || proposed.z>=8.0)) counts.w=6u;
  b_corrections[slot*2u]=vec4(correction,0.0);b_corrections[slot*2u+1u]=uintBitsToFloat(counts);
}
