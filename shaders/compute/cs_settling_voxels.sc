#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RW(b_support,vec4,1);
BUFFER_RO(b_world,uvec4,2);
uniform vec4 u_settling;
void voxelSupport(inout vec4 support,float gap) {
  if(gap< -0.00625)return;
  support.x=max(support.x,gap<=.0001?1.0:0.0);
  support.y=max(support.y,gap<=.002?1.0:0.0);
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_settling.x))return;
  vec4 support=b_support[id];
  // Pair pass validates query geometry and marks eligible dynamic solids.
  if(support.w==0.0 || floatBitsToUint(support.z)!=0u)return;
  vec3 p=b_state[id*4u].xyz;
  ivec3 minimum=ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
  vec3 lower=vec3(minimum)*.125,upper=lower+vec3(dimensions)*4.0;
  if(any(lessThan(p,lower+vec3_splat(.0625))) || any(greaterThan(p,upper-vec3_splat(.0625)))) {
    support.z=uintBitsToFloat(16u);b_support[id]=support;return;
  }
  // The resident floor is a containment plane even when its voxels are empty.
  voxelSupport(support,p.y-lower.y-.0625);
  // Hold shell spans at most three voxels per axis. Include closed tangencies.
  ivec3 first=max(ivec3(ceil((p-vec3_splat(.0645))*8.0))-ivec3(1,1,1)-minimum,ivec3(0,0,0));
  ivec3 last=min(ivec3(floor((p+vec3_splat(.0645))*8.0))-minimum,dimensions*32-ivec3(1,1,1));
  LOOP for(int z=first.z;z<=last.z;++z)
  LOOP for(int y=first.y;y<=last.y;++y)
  LOOP for(int x=first.x;x<=last.x;++x) {
    ivec3 cell=ivec3(x,y,z),chunk=cell/32,local=cell-chunk*32;
    uint address=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z))*32768u+uint(local.x+32*(local.y+32*local.z));
    uvec4 packed=b_world[4u+address/4u];if(packed[address%4u]==0xffffffffu)continue;
    vec3 lo=vec3(minimum+cell)*.125;
    vec3 delta=p-clamp(p,lo,lo+vec3_splat(.125));float distance=length(delta);
    // Closest face/edge/corner supplies the outward support normal. A centre
    // inside a voxel has no valid upward normal and cannot acquire from it.
    if(distance>1e-6 && delta.y>=.5*distance)voxelSupport(support,distance-.0625);
  }
  b_support[id]=support;
}
