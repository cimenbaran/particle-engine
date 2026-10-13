#include <bgfx_compute.sh>
BUFFER_RO(b_total,uint,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_debris;
NUM_THREADS(1,1,1)
void main() {
  uint requested=floatBitsToUint(u_debris.y);b_status[2]=b_total[0];
  if(requested>b_total[0])b_status[0]|=1u;
  b_status[3]=b_status[0]==0u?requested:0u;
}
