#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_state, vec4, 0);
BUFFER_RW(b_instances, vec4, 1);
BUFFER_RW(b_counts, uint, 2);
uniform vec4 u_cosmetic;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){
 uint id=gl_GlobalInvocationID.x;if(id>=uint(u_cosmetic.x))return;
 uint base=id*4u;uvec4 identity=floatBitsToUint(b_state[base+3u]);
 vec4 life=b_state[base+2u];life.w+=u_cosmetic.z;
 if(identity.y==0u || life.w>=life.z){
  identity.y=0u;b_state[base+3u]=uintBitsToFloat(identity);
  b_instances[id*2u]=vec4_splat(0.0);b_instances[id*2u+1u]=vec4_splat(0.0);return;
 }
 vec4 p=b_state[base],v=b_state[base+1u];bool spark=identity.x==1u;
 // Visual trajectories have no collision, force, heat or mass coupling.
 v.y+=(spark?-9.81:0.4)*u_cosmetic.z;
 v.xyz*=max(0.0,1.0-(spark?0.3:1.8)*u_cosmetic.z);
 p.xyz+=v.xyz*u_cosmetic.z;
 b_state[base]=p;b_state[base+1u]=v;b_state[base+2u]=life;
 float fade=clamp(1.0-life.w/life.z,0.0,1.0);
 uint alpha=uint((spark?255.0:120.0)*fade);
 uint color=(spark?0x0033aaffu:0x00808f9eu)|(alpha<<24u);
 b_instances[id*2u]=p;b_instances[id*2u+1u]=uintBitsToFloat(uvec4(color,0u,id,1u));
 uint ignored;atomicFetchAndAdd(b_counts[identity.x],1u,ignored);
}
