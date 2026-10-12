#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_corrected,vec4,0);
BUFFER_RO(b_candidate,vec4,1);
BUFFER_RO(b_status,uint,2);
BUFFER_RW(b_output,vec4,3);
uniform vec4 u_velocity;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x)) return;
  for(uint lane=0u;lane<4u;++lane) {
    uint index=id*4u+lane;
    if(b_status[0]==0u) b_output[index]=b_candidate[index];else b_output[index]=b_corrected[index];
  }
}
