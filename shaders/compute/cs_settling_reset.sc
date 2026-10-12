#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_history,vec4,0);
BUFFER_RW(b_status,uint,1);
BUFFER_RW(b_latched,uint,2);
uniform vec4 u_settling;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id==0u){b_status[0]=0u;b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;b_latched[0]=0u;b_latched[1]=0u;b_latched[2]=0u;b_latched[3]=0u;}
  if(id<uint(u_settling.x))b_history[id]=vec4_splat(0.0);
}
