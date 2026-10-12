#include <bgfx_compute.sh>
BUFFER_RO(b_counts, uint, 0);
IMAGE2D_RW(o_stats, rgba32ui, 1);
NUM_THREADS(1,1,1)
void main() {
  imageStore(o_stats,ivec2(0,0),uvec4(b_counts[0],b_counts[1],b_counts[2],b_counts[3]));
}
