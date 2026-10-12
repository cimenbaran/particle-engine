#include <bgfx_compute.sh>
#include "materials.sh"
BUFFER_RO(b_upload, vec4, 0);
BUFFER_RW(b_state, vec4, 1);
BUFFER_RW(b_instances, vec4, 2);
uniform vec4 u_sim;
NUM_THREADS(128, 1, 1)
void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= uint(u_sim.y)) return;
  uint base = id*4u;
  b_state[base] = b_upload[base];
  b_state[base+1u] = b_upload[base+1u];
  b_state[base+2u] = b_upload[base+2u];
  b_state[base+3u] = b_upload[base+3u];
  b_instances[id*2u] = b_upload[base];
  uvec4 identity=floatBitsToUint(b_upload[base+3u]);
  if(identity.x>=uint(u_sim.z) || (identity.y&1u)==0u) {b_instances[id*2u]=vec4(0.0); b_instances[id*2u+1u]=vec4(0.0); return;}
  b_instances[id*2u+1u]=uintBitsToFloat(uvec4(materialColor(identity.x),floatBitsToUint(b_materials[identity.x*4u+3u]).z,identity.z,identity.y));
}
