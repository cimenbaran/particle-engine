#include <bgfx_compute.sh>
BUFFER_RO(b_world,uvec4,0);
BUFFER_RW(b_status,uint,1);
NUM_THREADS(1,1,1)
void main() {
  // Dispatch/readback completion acknowledges the ordered immutable upload.
  b_status[0]=b_world[2].x;b_status[1]=b_world[2].y;
  b_status[2]=b_world[0].w;b_status[3]=b_world[1].w;
}
