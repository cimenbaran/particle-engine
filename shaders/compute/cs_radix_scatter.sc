#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_input,uint,0);
BUFFER_RO(b_flags,uint,1);
BUFFER_RO(b_prefix,uint,2);
BUFFER_RO(b_total,uint,3);
BUFFER_RW(b_output,uint,4);
uniform vec4 u_radix;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=uint(u_radix.x)) return;
  uint target=b_flags[id]!=0u?b_prefix[id]:b_total[0]+id-b_prefix[id];
  b_output[target*2u]=b_input[id*2u];
  b_output[target*2u+1u]=b_input[id*2u+1u];
}
