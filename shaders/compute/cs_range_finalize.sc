#include <bgfx_compute.sh>
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_meta,uint,1);
BUFFER_RO(b_total,uint,2);
uniform vec4 u_ranges;
NUM_THREADS(1,1,1)
void main() {
  uint observed=b_meta[1],ranges=b_total[0];
  if(observed!=b_source[0] || ranges>observed || observed>uint(u_ranges.x)) b_meta[2]|=16u;
  b_meta[3]=ranges;
  b_meta[0]=b_meta[2]==0u?ranges:0u;
  b_meta[1]=b_meta[2]==0u?observed:0u;
}
