#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_support,vec4,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_settling;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){uint id=gl_GlobalInvocationID.x;if(id<uint(u_settling.x))atomicOr(b_status[0],floatBitsToUint(b_support[id].z));}
