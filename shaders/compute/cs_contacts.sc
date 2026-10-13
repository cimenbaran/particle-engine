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
uniform vec4 u_contact_bounds; // optional inset boundary projection
#include "contact_math.sh"
bool validContactSlot(uint slot) {
  return b_state[slot*8u].w==u_contacts.y && validMass(b_state[slot*8u+1u].w);
}
void contact(uint queryIndex,uint slot,vec3 delta,float distanceSquared) {
  uint query=b_sorted[queryIndex*2u+1u];
  vec4 counts=b_corrections[query*2u+1u];
  if(!validContactSlot(slot)) {counts.w=uintBitsToFloat(4u);b_corrections[query*2u+1u]=counts;return;}
  if(b_state[slot*8u+5u].z!=0.0){counts.w=uintBitsToFloat(7u);b_corrections[query*2u+1u]=counts;return;}
  float distance=sqrt(distanceSquared),penetration=2.0*u_contacts.y-distance;
  if(penetration<=0.0) return;
  counts.x=uintBitsToFloat(floatBitsToUint(counts.x)+1u);b_corrections[query*2u+1u]=counts;
  float a=b_state[query*8u+1u].w,b=b_state[slot*8u+1u].w;
  if(a==0.0) return;
  float weight=massWeight(a,b);
  vec3 direction=contactNormal(query,slot,delta,distanceSquared);
  vec3 change=direction*penetration;
  if(b_state[query*8u+5u].x!=0.0 && distanceSquared>1e-12) {
    vec3 relative=(b_state[query*8u].xyz-b_state[query*8u+4u].xyz)-(b_state[slot*8u].xyz-b_state[slot*8u+4u].xyz);
    vec3 tangent=relative-direction*dot(relative,direction);
    float length=sqrt(dot(tangent,tangent)),mu=sqrt(b_state[query*8u+5u].y*b_state[slot*8u+5u].y);
    // Static traction only. Sliding friction stays in the velocity response,
    // avoiding a second kinetic-friction impulse for the same contact.
    if(length<=mu*penetration)change-=tangent;
  }
  vec4 correction=b_corrections[query*2u];correction.xyz+=change*weight;
  b_corrections[query*2u]=correction;
}
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) contact(_query,_slot,_delta,_distance)
#define PW_STATE_STRIDE 8u
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
  if(b_sorted[id*2u]!=0xffffffffu && b_state[slot*8u+5u].z!=0.0)result=uvec4(0u,0u,0u,7u);
  else if(b_sorted[id*2u]!=0xffffffffu && !validContactSlot(slot)) result=uvec4(0u,0u,0u,4u);
  else result=visitNeighbors(id,u_neighbors);
  uvec4 counts=floatBitsToUint(b_corrections[slot*2u+1u]);
  counts.y=result.y;counts.z=result.z;counts.w=max(counts.w,result.w);
  vec3 correction=b_corrections[slot*2u].xyz*u_contacts.z;
  float lengthSquared=dot(correction,correction);
  if(lengthSquared>u_contacts.w*u_contacts.w) correction*=u_contacts.w/sqrt(lengthSquared);
  if((floatBitsToUint(correction.x)&0x7f800000u)==0x7f800000u ||
     (floatBitsToUint(correction.y)&0x7f800000u)==0x7f800000u ||
     (floatBitsToUint(correction.z)&0x7f800000u)==0x7f800000u) counts.w=5u;
  vec3 position=b_state[slot*8u].xyz,proposed=position+correction;
  if(u_contact_bounds.x!=0.0 && counts.w==0u && b_state[slot*8u+1u].w!=0.0) {
    vec3 lo=vec3(-8.0,0.0,-8.0)+vec3_splat(u_contacts.y);
    vec3 hi=vec3(8.0,16.0,8.0)-vec3_splat(u_contacts.y);
    // A contained input makes this projection non-expansive: the correction
    // still satisfies the configured per-iteration displacement bound.
    if(any(lessThan(position,lo)) || any(greaterThan(position,hi))) counts.w=6u;
    else {
      vec3 projected=clamp(proposed,lo,hi),normalDelta=projected-proposed;
      if(b_state[slot*8u+5u].x!=0.0) {
        vec3 tangent=projected-b_state[slot*8u+4u].xyz;
        // Simultaneous planes share only their common tangent axes.
        if(normalDelta.x!=0.0)tangent.x=0.0;
        if(normalDelta.y!=0.0)tangent.y=0.0;
        if(normalDelta.z!=0.0)tangent.z=0.0;
        float remainingFloor=0.0;
        if(position.y==lo.y && b_state[slot*8u+4u].y==lo.y && normalDelta.x==0.0 && normalDelta.z==0.0) {
          remainingFloor=b_state[slot*8u+6u].x;
          if(remainingFloor>0.0)tangent.y=0.0;
        }
        if(length(tangent)<=b_state[slot*8u+5u].y*length(normalDelta)+remainingFloor)projected-=tangent;
      }
      proposed=clamp(projected,lo,hi);correction=proposed-position;
      // Friction may restore an earlier substep position. Bound the combined
      // update again; interpolation from the contained input stays contained.
      float combined=dot(correction,correction);
      if(combined>u_contacts.w*u_contacts.w)correction*=u_contacts.w/sqrt(combined);
      proposed=position+correction;
    }
  }
  if(counts.w==0u && (proposed.x< -8.0 || proposed.x>=8.0 || proposed.y<0.0 || proposed.y>=16.0 || proposed.z< -8.0 || proposed.z>=8.0)) counts.w=6u;
  b_corrections[slot*2u]=vec4(correction,0.0);b_corrections[slot*2u+1u]=uintBitsToFloat(counts);
}
