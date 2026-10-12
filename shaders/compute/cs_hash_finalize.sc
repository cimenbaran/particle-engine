#include <bgfx_compute.sh>
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_meta,uint,1);
NUM_THREADS(1,1,1)
void main() {
  if(b_meta[3]==0u && (b_meta[1]!=b_source[0] || b_meta[2]!=0u)) b_meta[3]=32u;
  b_meta[0]=b_meta[3]==0u?1u:0u;
}
