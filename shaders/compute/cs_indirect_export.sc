#include <bgfx_compute.sh>
// D3D11 indirect buffers expose a uint4 UAV, without an SRV. Read through
// that existing view; no stores to the argument buffer are performed.
BUFFER_RW(b_args, uvec4, 0);
UIMAGE2D_WO(t_result, rgba32ui, 1);
NUM_THREADS(1,1,1)
void main() {
  imageStore(t_result, ivec2(0,0), b_args[0]);
  imageStore(t_result, ivec2(1,0), b_args[1]);
}
