#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_owners,uint,0);
BUFFER_RW(b_result,uint,1);
uniform vec4 u_audit; // capacity, material count, clear latched errors
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;
  if(slot==0u) {
    b_result[0]=0u;
    if(u_audit.z>0.0) {b_result[1]=0u;b_result[2]=0u;b_result[3]=0u;}
  }
  if(slot<uint(u_audit.x)) b_owners[slot]=0u;
}
