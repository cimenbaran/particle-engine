#include <bgfx_compute.sh>
BUFFER_RO(b_reservation,uint,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_debris_activation;
NUM_THREADS(1,1,1)
void main() {
  uint capacity=uint(u_debris_activation.x),count=uint(u_debris_activation.y);
  b_status[0]=b_reservation[0]!=0u || count>capacity || b_reservation[1]!=count ||
    b_reservation[2]>capacity || b_reservation[3]!=count || b_reservation[3]>b_reservation[2]?1u:0u;
  b_status[1]=count;b_status[2]=0u;b_status[3]=0u;
}
