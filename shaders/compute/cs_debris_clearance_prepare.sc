#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_claims,uvec4,1);
BUFFER_RO(b_activation,uint,2);
BUFFER_RO(b_materials,vec4,3);
BUFFER_RW(b_queries,vec4,4);
uniform vec4 u_debris_clearance;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_debris_clearance.y))return;
  uint error=0u;uvec4 claim=uvec4_splat(0u);vec4 pose=vec4_splat(0.0);
  if(b_activation[0]!=0u || b_activation[1]!=uint(u_debris_clearance.y) || b_activation[3]!=uint(u_debris_clearance.y) || id>=uint(u_debris_clearance.x))error=1u;
  else {
    claim=b_claims[id];
    if(claim.x>=uint(u_debris_clearance.x) || claim.y==0u || claim.z==0xffffffffu || claim.y!=claim.z+1u || claim.w!=0u)error=2u;
    else {
      uvec4 identity=floatBitsToUint(b_state[claim.x*4u+3u]);pose=b_state[claim.x*4u];
      if(identity.y!=1u || identity.z!=claim.x || identity.w!=claim.y || identity.x>=uint(u_debris_clearance.z))error=2u;
      else if(floatBitsToUint(b_materials[identity.x*4u+3u]).y!=0u)error=2u;
      if(id>0u && b_claims[id-1u].x>=claim.x)error=2u;
    }
  }
  b_queries[id*3u]=pose;b_queries[id*3u+1u]=uintBitsToFloat(claim);b_queries[id*3u+2u]=uintBitsToFloat(uvec4(error,0u,0u,0u));
}
