#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_current,vec4,0);
BUFFER_RO(b_start,vec4,1);
BUFFER_RO(b_materials,vec4,2);
BUFFER_RW(b_packed,vec4,3);
uniform vec4 u_contact_friction; // capacity, material count, static friction enabled
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_contact_friction.x))return;
  for(uint lane=0u;lane<4u;++lane)b_packed[id*8u+lane]=b_current[id*4u+lane];
  for(uint lane=4u;lane<8u;++lane)b_packed[id*8u+lane]=vec4_splat(0.0);
  if(u_contact_friction.z==0.0)return;
  uvec4 identity=floatBitsToUint(b_current[id*4u+3u]);
  if((identity.y&1u)==0u || identity.x>=uint(u_contact_friction.y))return;
  if(floatBitsToUint(b_materials[identity.x*4u+3u]).y!=0u)return;
  uvec4 old=floatBitsToUint(b_start[id*4u+3u]);vec4 p=b_start[id*4u];
  float mu=b_materials[identity.x*4u].y;uint bits=floatBitsToUint(mu),magnitude=bits&0x7fffffffu;
  bool valid=all(equal(identity,old)) && contactFinite3(p.xyz) &&
    all(greaterThanEqual(p.xyz,vec3(-8.0,0.0,-8.0))) && all(lessThan(p.xyz,vec3(8.0,16.0,8.0))) &&
    p.w==b_current[id*4u].w && floatBitsToUint(b_start[id*4u+1u].w)==floatBitsToUint(b_current[id*4u+1u].w) &&
    magnitude<=0x3f800000u && ((bits&0x80000000u)==0u || magnitude==0u);
  b_packed[id*8u+4u]=p;
  // enabled, material friction, validation error, reserved
  b_packed[id*8u+5u]=vec4(1.0,mu,valid?0.0:1.0,0.0);
}
