#include "contact_math.sh"
BUFFER_RO(b_input,vec4,0);
BUFFER_RO(b_materials,vec4,1);
BUFFER_RO(b_upstream,uint,2);
BUFFER_RW(b_candidate,vec4,3);
BUFFER_RW(b_status,uint,4);
uniform vec4 u_velocity;
uniform vec4 u_granular; // gravity, speed limit
// Return 0 for excluded, 1 dynamic, 2 static, 3 invalid.
uint granularSlot(uint id) {
  uvec4 identity=floatBitsToUint(b_input[id*4u+3u]);if((identity.y&1u)==0u)return 0u;
  if(identity.x>=uint(u_velocity.z))return 3u;
  uint phase=floatBitsToUint(b_materials[identity.x*4u+3u]).y;
  if(phase>2u)return 3u;if(phase!=0u)return 0u;
  vec4 pose=b_input[id*4u],motion=b_input[id*4u+1u],thermal=b_input[id*4u+2u];
  if(identity.z!=id || identity.w==0u || pose.w!=0.0625 || !validMass(motion.w) ||
     !contactFinite3(pose.xyz) || !contactFinite3(motion.xyz) || !contactFinite3(thermal.xyz) ||
     !contactFinite(thermal.w) || thermal.z<0.0 || thermal.w<0.0)return 3u;
  return motion.w==0.0?2u:1u;
}
