#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state, vec4, 0);
UIMAGE2D_WO(t_result, rgba32ui, 1);
uniform vec4 u_sim;
NUM_THREADS(PW_WORKGROUP_SIZE, 1, 1)
void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= uint(u_sim.y)) return;
  uint lanes=u_sim.w>0.0?uint(u_sim.w):4u;
  for (uint lane = 0u; lane < lanes; ++lane)
    imageStore(t_result, ivec2(int(lane),int(id)), floatBitsToUint(b_state[id*lanes+lane]));
}
