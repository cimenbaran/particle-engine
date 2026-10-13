#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
#include "voxel_contact.sh"
BUFFER_RO(b_segments,vec4,0);
BUFFER_RW(b_results,vec4,1);
uniform vec4 u_voxel_query; // lower corner xyz, capacity
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_voxel_query.w))return;
  vec4 start=b_segments[id*2u],end=b_segments[id*2u+1u];
  if(!contactFinite3(start.xyz) || !contactFinite3(end.xyz) || start.w!=0.0625 || end.w!=0.0 ||
     any(greaterThan(abs(start.xyz),vec3_splat(64.0))) || any(greaterThan(abs(end.xyz),vec3_splat(64.0)))) {
    b_results[id*2u]=vec4(0.0,0.0,0.0,1.0);b_results[id*2u+1u]=vec4(0.0,0.0,0.0,uintBitsToFloat(3u));return;
  }
  vec3 centre,normal;float time;
  uint kind=voxelSweep(start.xyz,end.xyz,u_voxel_query.xyz,centre,normal,time);
  if(!contactFinite3(centre) || !contactFinite3(normal) || !contactFinite(time) || time<0.0 || time>1.0) {
    kind=3u;centre=vec3_splat(0.0);normal=vec3_splat(0.0);time=1.0;
  }
  b_results[id*2u]=vec4(normal,time);b_results[id*2u+1u]=vec4(centre,uintBitsToFloat(kind));
}
