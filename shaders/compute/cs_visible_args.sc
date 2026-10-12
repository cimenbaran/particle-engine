#include <bgfx_compute.sh>
BUFFER_RO(b_counts, uint, 0);
BUFFER_RW(b_args, uvec4, 1);
uniform vec4 u_cull;
NUM_THREADS(1,1,1)
void main() {
  // One indexed quad. Both static geometry buffers start at element zero.
  b_args[0]=uvec4(6u,min(b_counts[0],uint(u_cull.x)),0u,0u);
  b_args[1]=uvec4(0u);
}
