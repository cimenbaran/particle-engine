#include <bgfx_compute.sh>
BUFFER_RW(b_counts, uint, 0);
uniform vec4 u_pool; // capacity, unused, unused, reset diagnostics
NUM_THREADS(1,1,1)
void main() {
  b_counts[0]=0u; b_counts[1]=0u; b_counts[2]=0u;
  if(u_pool.w>0.0) b_counts[3]=0u;
}
