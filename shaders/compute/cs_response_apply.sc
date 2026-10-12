#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_current,vec4,0);
BUFFER_RO(b_result,vec4,1);
BUFFER_RO(b_packed,vec4,2);
BUFFER_RO(b_status,uint,3);
BUFFER_RW(b_output,vec4,4);
uniform vec4 u_response;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_response.x)) return;
  uint base=id*4u;for(uint i=0u;i<4u;++i) b_output[base+i]=b_current[base+i];
  if(b_status[0]==0u && floatBitsToUint(b_packed[id*8u+6u].y)==1u && b_current[base+1u].w!=0.0)
    b_output[base+1u]=vec4(b_result[id*2u].xyz,b_current[base+1u].w);
}
