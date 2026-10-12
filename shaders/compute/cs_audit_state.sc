#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_owners,uint,1);
BUFFER_RW(b_result,uint,2);
uniform vec4 u_audit;
// Exponent bits detect NaN/Inf without depending on fast-math comparisons.
bool finiteComponent(float value) {return (floatBitsToUint(value)&0x7f800000u)!=0x7f800000u;}
bool finiteLane(vec4 value) {
  return finiteComponent(value.x) && finiteComponent(value.y) &&
    finiteComponent(value.z) && finiteComponent(value.w);
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;
  if(slot>=uint(u_audit.x)) return;
  atomicAdd(b_result[0],1u);
  if(b_owners[slot]!=1u) atomicOr(b_result[1],1u);
  vec4 position=b_state[slot*4u],velocity=b_state[slot*4u+1u],thermal=b_state[slot*4u+2u];
  uvec4 identity=floatBitsToUint(b_state[slot*4u+3u]);
  bool alive=(identity.y&1u)!=0u;
  bool valid=finiteLane(position) && finiteLane(velocity) && finiteLane(thermal);
  if(alive) {
    valid=valid && position.w>0.0 && position.w<8.0 && velocity.w>0.0 &&
      position.y>=position.w-0.001 && abs(position.x)<=8.0-position.w+0.001 &&
      abs(position.z)<=8.0-position.w+0.001 && thermal.x>=0.0 &&
      thermal.y>=0.0 && thermal.y<=1.0 && thermal.z>=0.0 && thermal.w>=0.0;
  }
  if(!valid) atomicOr(b_result[2],1u);
  if(identity.z!=slot || (alive && (identity.w==0u || identity.x>=uint(u_audit.y)))) atomicOr(b_result[3],1u);
}
