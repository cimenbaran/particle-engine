#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "granular_io.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x))return;
  uint base=id*4u;for(uint lane=0u;lane<4u;++lane)b_candidate[base+lane]=b_input[base+lane];
  if(b_upstream[0]!=0u)return;
  uint kind=granularSlot(id);if(kind==0u)return;if(kind==3u){atomicOr(b_status[0],1u);return;}
  vec4 thermal=b_input[base+2u];thermal.w+=u_velocity.y;
  if(!contactFinite(thermal.w)){atomicOr(b_status[0],1u);return;}
  b_candidate[base+2u]=thermal;
  if(thermal.z>0.0 && thermal.w>=thermal.z) {
    uvec4 identity=floatBitsToUint(b_input[base+3u]);identity.y&=~1u;b_candidate[base+3u]=uintBitsToFloat(identity);return;
  }
  if(kind==2u)return;
  uint material=floatBitsToUint(b_input[base+3u]).x;float drag=b_materials[material*4u].w;
  if(!contactFinite(drag) || drag<0.0 || drag>100.0){atomicOr(b_status[0],1u);return;}
  vec4 motion=b_input[base+1u];motion.y-=u_granular.x*u_velocity.y;
  motion.xyz*=max(0.0,1.0-drag*u_velocity.y);
  if(!contactFinite3(motion.xyz)){atomicOr(b_status[0],1u);return;}
  b_candidate[base+1u]=motion;
}
