#include <bgfx_compute.sh>
BUFFER_RO(b_health,uint,0);
BUFFER_RO(b_clearance,uint,1);
BUFFER_WO(b_status,uint,2);
NUM_THREADS(1,1,1)
void main(){b_status[0]=b_clearance[0]|(b_health[0]!=0u?32u:0u);b_status[1]=b_clearance[1];b_status[2]=b_clearance[2];b_status[3]=b_clearance[3];}
