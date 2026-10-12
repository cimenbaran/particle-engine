#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_previous,vec4,0);
BUFFER_RO(b_corrected,vec4,1);
BUFFER_RO(b_materials,vec4,2);
BUFFER_RO(b_upstream,uint,3);
BUFFER_RW(b_candidate,vec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_velocity; // capacity, dt, material count, reserved
bool finiteValue(float v) {return (floatBitsToUint(v)&0x7f800000u)!=0x7f800000u;}
bool finitePosition(vec3 v) {return finiteValue(v.x) && finiteValue(v.y) && finiteValue(v.z);}
float liftedTinyCoordinate(float value) {
  uint bits=floatBitsToUint(value);
  if((bits&0x7f800000u)==0u) {
    float magnitude=float(bits&0x007fffffu)*2.5849394142282115e-26;
    return (bits&0x80000000u)!=0u?-magnitude:magnitude;
  }
  return value*18446744073709551616.0;
}
float reconstructComponent(float before,float after,float dt) {
  // Preserve subnormal separations that can yield normal velocities at small
  // timesteps. Both scaled coordinates and scaled dt remain finite and normal.
  if(max(floatBitsToUint(before)&0x7fffffffu,floatBitsToUint(after)&0x7fffffffu)<0x20000000u)
    return (liftedTinyCoordinate(after)-liftedTinyCoordinate(before))/(dt*18446744073709551616.0);
  return (after-before)/dt;
}
void fail(uint error) {atomicOr(b_status[0],error);atomicAdd(b_status[3],1u);}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x)) return;
  uint base=id*4u;
  for(uint i=0u;i<4u;++i) b_candidate[base+i]=b_corrected[base+i];
  if(b_upstream[0]!=0u) return;
  uvec4 identity=floatBitsToUint(b_corrected[base+3u]);
  if((identity.y&1u)==0u) {atomicAdd(b_status[2],1u);return;}
  if(identity.x>=uint(u_velocity.z)) {fail(4u);return;}
  uint phase=floatBitsToUint(b_materials[identity.x*4u+3u]).y;
  if(phase>2u) {fail(4u);return;}
  if(phase!=0u) {atomicAdd(b_status[2],1u);return;}
  uvec4 oldIdentity=floatBitsToUint(b_previous[base+3u]);
  if(any(notEqual(identity,oldIdentity)) || identity.z!=id || identity.w==0u ||
     floatBitsToUint(b_previous[base].w)!=floatBitsToUint(b_corrected[base].w) ||
     floatBitsToUint(b_previous[base+1u].w)!=floatBitsToUint(b_corrected[base+1u].w)) {fail(2u);return;}
  vec3 before=b_previous[base].xyz,after=b_corrected[base].xyz;
  float mass=b_corrected[base+1u].w;uint massBits=floatBitsToUint(mass),magnitude=massBits&0x7fffffffu;
  if(!finitePosition(before) || !finitePosition(after) || !finiteValue(mass) ||
     (magnitude!=0u && ((massBits&0x80000000u)!=0u || magnitude<0x00800000u))) {fail(4u);return;}
  if(magnitude==0u) {atomicAdd(b_status[2],1u);return;}
  vec3 velocity=vec3(reconstructComponent(before.x,after.x,u_velocity.y),
    reconstructComponent(before.y,after.y,u_velocity.y),reconstructComponent(before.z,after.z,u_velocity.y));
  if(!finitePosition(velocity)) {fail(8u);return;}
  b_candidate[base+1u]=vec4(velocity,mass);atomicAdd(b_status[1],1u);
}
