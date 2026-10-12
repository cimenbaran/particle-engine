#include <bgfx_compute.sh>
BUFFER_RO(b_upstream,uint,0);
BUFFER_RW(b_status,uint,1);
NUM_THREADS(1,1,1)
void main() {b_status[0]=b_upstream[0]==0u?0u:1u;b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;}
