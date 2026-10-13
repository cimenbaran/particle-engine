#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_segments,vec4,0);
BUFFER_RW(b_results,vec4,1);
BUFFER_RO(b_world,uvec4,2);
#include "voxel_world_contact.sh"
uniform vec4 u_voxel_world_query;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_voxel_world_query.x))return;
  vec4 start=b_segments[id*2u],end=b_segments[id*2u+1u];
  vec3 centre=vec3_splat(0.0),normal=vec3_splat(0.0);float time=1.0;
  uint material=0xffffffffu,index=0xffffffffu,kind=3u;
  if(start.w==0.0625 && end.w==0.0)kind=voxelWorldSweep(start.xyz,end.xyz,false,0xffffffffu,centre,normal,time,material,index);
  b_results[id*3u]=vec4(normal,time);b_results[id*3u+1u]=vec4(centre,uintBitsToFloat(kind));
  b_results[id*3u+2u]=uintBitsToFloat(uvec4(material,index,b_world[2].xy));
}
