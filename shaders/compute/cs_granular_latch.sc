#include <bgfx_compute.sh>
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_latched,uint,1);
NUM_THREADS(1,1,1)
void main() {
  b_latched[0]|=b_source[0];
  b_latched[1]=b_source[1];b_latched[2]=b_source[2];b_latched[3]=b_source[3];
}
