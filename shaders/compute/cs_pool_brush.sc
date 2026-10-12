#include <bgfx_compute.sh>
BUFFER_RW(b_state, vec4, 0);
uniform vec4 u_pool; // capacity, 0 impulse / 1 remove
uniform vec4 u_brushMin; // xyz, material ID
uniform vec4 u_brushMax;
uniform vec4 u_brushValue;
NUM_THREADS(128,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;
  if(slot>=uint(u_pool.x)) return;
  uint base=slot*4u;
  uvec4 identity=floatBitsToUint(b_state[base+3u]);
  vec3 p=b_state[base].xyz;
  if((identity.y&1u)==0u || identity.x!=uint(u_brushMin.w) ||
     any(lessThan(p,u_brushMin.xyz)) || any(greaterThan(p,u_brushMax.xyz))) return;
  if(u_pool.y>0.0) {
    identity.y=identity.y&~1u;
    b_state[base+3u]=uintBitsToFloat(identity);
  } else {
    vec4 velocity=b_state[base+1u]; velocity.xyz+=u_brushValue.xyz;
    b_state[base+1u]=velocity;
  }
}
