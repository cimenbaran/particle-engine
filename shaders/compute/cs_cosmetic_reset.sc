#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_state, vec4, 0);
BUFFER_RW(b_instances, vec4, 1);
BUFFER_RW(b_counts, uint, 2);
uniform vec4 u_cosmetic; // total, dust capacity, dt, seed count
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){
 uint id=gl_GlobalInvocationID.x;if(id>=uint(u_cosmetic.x))return;
 for(uint j=0u;j<4u;++j)b_state[id*4u+j]=vec4_splat(0.0);
 b_instances[id*2u]=vec4_splat(0.0);b_instances[id*2u+1u]=vec4_splat(0.0);
 if(id==0u)for(uint j=0u;j<4u;++j)b_counts[j]=0u;
}
