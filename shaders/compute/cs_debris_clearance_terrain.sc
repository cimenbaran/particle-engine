#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RW(b_queries,vec4,0);
BUFFER_RO(b_world,uvec4,1);
#include "contact_math.sh"
#include "voxel_world_contact.sh"
uniform vec4 u_debris_clearance;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_debris_clearance.y))return;
  uvec4 result=floatBitsToUint(b_queries[id*3u+2u]);if((result.x&7u)!=0u)return;
  vec3 p=b_queries[id*3u].xyz,centre,normal;float time;uint material,index;
  result.w=voxelWorldSweep(p,p,false,0xffffffffu,centre,normal,time,material,index);
  if(result.w>1u)result.x|=16u;b_queries[id*3u+2u]=uintBitsToFloat(result);
}
