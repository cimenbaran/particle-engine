#include <bgfx_compute.sh>
BUFFER_RW(b_counts, uint, 0);
NUM_THREADS(1,1,1)
void main(){b_counts[0]=0u;b_counts[1]=0u;}
