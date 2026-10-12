#include <bgfx_compute.sh>
BUFFER_RW(b_status,uint,0);
NUM_THREADS(1,1,1)
void main() {b_status[0]=0u;b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;}
