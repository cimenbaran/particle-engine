#include <bgfx_compute.sh>
#include "materials.sh"
BUFFER_RO(b_previous, vec4, 0);
BUFFER_RW(b_next, vec4, 1);
BUFFER_RW(b_instances, vec4, 2);
uniform vec4 u_sim; // dt, active count, material count, unused
NUM_THREADS(128, 1, 1)
void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= uint(u_sim.y)) return;
  uint base = id * 4u;
  uvec4 identity=floatBitsToUint(b_previous[base+3u]);
  if(identity.x>=uint(u_sim.z) || (identity.y&1u)==0u ||
     (b_previous[base+2u].z>0.0 && b_previous[base+2u].w+u_sim.x>=b_previous[base+2u].z)) {
    for(uint lane=0u;lane<4u;++lane) b_next[base+lane]=vec4(0.0);
    identity.y=identity.y&~1u; b_next[base+3u]=uintBitsToFloat(identity);
    b_instances[id*2u]=vec4(0.0); b_instances[id*2u+1u]=vec4(0.0);
    return;
  }
  vec4 material=b_materials[identity.x*4u];
  vec4 p = b_previous[base];
  vec4 v = b_previous[base+1u];
  v.y -= 9.81 * u_sim.x;
  v.xyz *= max(0.0, 1.0 - material.w*u_sim.x);
  p.xyz += v.xyz * u_sim.x;
  if (p.y < p.w) { p.y = p.w; v.y = abs(v.y)*material.z; v.xz *= 1.0-material.y; }
  if (abs(p.x) > 8.0-p.w) { p.x = sign(p.x)*(8.0-p.w); v.x *= -material.z; }
  if (abs(p.z) > 8.0-p.w) { p.z = sign(p.z)*(8.0-p.w); v.z *= -material.z; }
  b_next[base] = p;
  b_next[base+1u] = v;
  vec4 thermal = b_previous[base+2u]; thermal.w += u_sim.x;
  b_next[base+2u] = thermal;
  b_next[base+3u] = b_previous[base+3u];
  b_instances[id*2u] = p;
  b_instances[id*2u+1u] = uintBitsToFloat(uvec4(materialColor(identity.x), floatBitsToUint(b_materials[identity.x*4u+3u]).z, identity.z, identity.y));
}
