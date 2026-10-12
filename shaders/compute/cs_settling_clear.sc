#include <bgfx_compute.sh>
BUFFER_RO(b_upstream,uint,0);
BUFFER_RO(b_meta,uint,1);
BUFFER_RW(b_status,uint,2);
BUFFER_RO(b_latched,uint,3);
uniform vec4 u_settling;
NUM_THREADS(1,1,1)
void main(){b_status[0]=(b_upstream[0]!=0u?1u:0u)|(b_meta[0]!=1u?2u:0u)|(u_settling.w!=0.0?b_latched[0]:0u);b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;}
