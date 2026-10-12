#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_corrections,vec4,1);
BUFFER_RO(b_status,uint,2);
BUFFER_RW(b_output,vec4,3);
uniform vec4 u_contacts;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_contacts.x)) return;
  vec4 position=b_state[id*4u];
  if(b_status[0]==0u) {
    vec3 correction=b_corrections[id*2u].xyz;
    if(correction.x!=0.0 || correction.y!=0.0 || correction.z!=0.0) position.xyz+=correction;
  }
  b_output[id*4u]=position;
  b_output[id*4u+1u]=b_state[id*4u+1u];b_output[id*4u+2u]=b_state[id*4u+2u];b_output[id*4u+3u]=b_state[id*4u+3u];
}
