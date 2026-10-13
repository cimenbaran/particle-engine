#include <bgfx_compute.sh>
BUFFER_RW(b_status,uint,0);
NUM_THREADS(1,1,1)
void main(){for(uint i=0u;i<4u;++i)b_status[i]=0u;}
