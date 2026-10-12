#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_output, vec4, 0);
BUFFER_RW(b_counts, uint, 1);
uniform vec4 u_cull;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id==0u) {b_counts[0]=0u;b_counts[1]=0u;b_counts[2]=0u;b_counts[3]=0u;}
  if(id>=uint(u_cull.x)) return;
  b_output[id*2u]=vec4(0.0);b_output[id*2u+1u]=vec4(0.0);
}
