#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_queries,vec4,0);
BUFFER_RW(b_status,uint,1);
uniform vec4 u_debris_clearance;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){uint id=gl_GlobalInvocationID.x;if(id>=uint(u_debris_clearance.y))return;uint error=floatBitsToUint(b_queries[id*3u+2u]).x;atomicOr(b_status[0],error);if(error==0u)atomicAdd(b_status[2],1u);else atomicAdd(b_status[3],1u);}
