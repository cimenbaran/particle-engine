#include <bgfx_compute.sh>
BUFFER_RO(b_previous, vec4, 0);
BUFFER_RW(b_next, vec4, 1);
BUFFER_RW(b_instances, vec4, 2);
uniform vec4 u_sim; // dt, active count, unused, unused
NUM_THREADS(128, 1, 1)
void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= uint(u_sim.y)) return;
  uint base = id * 4u;
  vec4 p = b_previous[base];
  vec4 v = b_previous[base+1u];
  v.y -= 9.81 * u_sim.x;
  v.xyz *= max(0.0, 1.0 - 0.05*u_sim.x);
  p.xyz += v.xyz * u_sim.x;
  if (p.y < p.w) { p.y = p.w; v.y = abs(v.y)*0.55; v.xz *= 0.98; }
  if (abs(p.x) > 8.0-p.w) { p.x = sign(p.x)*(8.0-p.w); v.x *= -0.6; }
  if (abs(p.z) > 8.0-p.w) { p.z = sign(p.z)*(8.0-p.w); v.z *= -0.6; }
  b_next[base] = p;
  b_next[base+1u] = v;
  vec4 thermal = b_previous[base+2u]; thermal.w += u_sim.x;
  b_next[base+2u] = thermal;
  b_next[base+3u] = b_previous[base+3u];
  b_instances[id*2u] = p;
  b_instances[id*2u+1u] = uintBitsToFloat(uvec4(0xff388ed9u, 0u, id, 1u));
}
