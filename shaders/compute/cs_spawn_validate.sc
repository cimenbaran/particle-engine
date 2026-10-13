#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_keys,uint,1);
BUFFER_RW(b_status,uint,2);
uniform vec4 u_spawn_index;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_spawn_index.x))return;
  // Validate every indexed solid, even outside all candidate neighborhoods.
  if(b_keys[id*2u]!=0xffffffffu && floatBitsToUint(b_state[id*4u].w)!=0x3d800000u)atomicAdd(b_status[2],1u);
}
