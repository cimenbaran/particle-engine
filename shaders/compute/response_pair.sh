#include "contact_math.sh"
// Packed state stride eight: corrected state, predicted pose/velocity,
// immutable validation/material info, then per-body degree diagnostics.
bool responsePair(uint a,uint b,float currentSquared) {
  if(b_state[a*8u+1u].w==0.0 && b_state[b*8u+1u].w==0.0) return false;
  vec3 oldDelta=b_state[b*8u+4u].xyz-b_state[a*8u+4u].xyz;
  return dot(oldDelta,oldDelta)<=0.015625 || currentSquared<=(0.125+u_response.z)*(0.125+u_response.z);
}
