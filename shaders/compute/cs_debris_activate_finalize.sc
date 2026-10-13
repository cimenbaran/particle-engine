#include <bgfx_compute.sh>
BUFFER_RW(b_status,uint,0);
uniform vec4 u_debris_activation;
NUM_THREADS(1,1,1)
void main() {
  if(b_status[2]!=uint(u_debris_activation.y))b_status[0]|=2u;
  b_status[3]=b_status[0]==0u?uint(u_debris_activation.y):0u;
}
