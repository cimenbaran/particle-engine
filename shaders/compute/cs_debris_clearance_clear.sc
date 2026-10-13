#include <bgfx_compute.sh>
BUFFER_RO(b_activation,uint,0);
BUFFER_RO(b_meta,uint,1);
BUFFER_RW(b_status,uint,2);
uniform vec4 u_debris_clearance;
NUM_THREADS(1,1,1)
void main(){b_status[0]=b_activation[0]!=0u || b_activation[1]!=uint(u_debris_clearance.y) || b_activation[3]!=uint(u_debris_clearance.y)?1u:0u;if(b_meta[0]!=1u)b_status[0]|=4u;b_status[1]=uint(u_debris_clearance.y);b_status[2]=0u;b_status[3]=0u;}
