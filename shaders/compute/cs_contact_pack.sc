#include "granular_grid.sh"
#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_current,vec4,0);
BUFFER_RO(b_start,vec4,1);
BUFFER_RO(b_materials,vec4,2);
BUFFER_RW(b_packed,vec4,3);
uniform vec4 u_contact_friction; // capacity, material count, static friction enabled
uniform vec4 u_contact_floor; // gravity, substep seconds, speed limit
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_contact_friction.x))return;
  for(uint lane=0u;lane<4u;++lane)b_packed[id*8u+lane]=b_current[id*4u+lane];
  for(uint clearLane=4u;clearLane<8u;++clearLane)b_packed[id*8u+clearLane]=vec4_splat(0.0);
  if(u_contact_friction.z==0.0)return;
  uvec4 identity=floatBitsToUint(b_current[id*4u+3u]);
  if((identity.y&1u)==0u || identity.x>=uint(u_contact_friction.y))return;
  if(floatBitsToUint(b_materials[identity.x*4u+3u]).y!=0u)return;
  uvec4 old=floatBitsToUint(b_start[id*4u+3u]);vec4 p=b_start[id*4u];
  float mu=b_materials[identity.x*4u].y;uint bits=floatBitsToUint(mu),magnitude=bits&0x7fffffffu;
  bool valid=all(equal(identity,old)) && contactFinite3(p.xyz) &&
    all(greaterThanEqual(p.xyz,vec3(-PW_GRID_EXTENT,0.0,-PW_GRID_EXTENT))) && all(lessThan(p.xyz,vec3(PW_GRID_EXTENT,16.0,PW_GRID_EXTENT))) &&
    p.w==b_current[id*4u].w && floatBitsToUint(b_start[id*4u+1u].w)==floatBitsToUint(b_current[id*4u+1u].w) &&
    magnitude<=0x3f800000u && ((bits&0x80000000u)==0u || magnitude==0u);
  if(u_contact_floor.x>0.0) {
    vec3 velocity=b_start[id*4u+1u].xyz;float drag=b_materials[identity.x*4u].w;
    valid=valid && contactFinite3(velocity) && contactFinite(drag) && drag>=0.0 && drag<=100.0;
    if(valid && p.y==p.w && abs(velocity.y)<=u_contact_floor.z && abs(velocity.x)<=u_contact_floor.z && abs(velocity.z)<=u_contact_floor.z) {
      float damping=max(0.0,1.0-drag*u_contact_floor.y);
      float tangent=length(velocity.xz)*damping,normal=max(0.0,u_contact_floor.x*u_contact_floor.y-velocity.y)*damping;
      float speed=sqrt(tangent*tangent+normal*normal);
      if(speed>u_contact_floor.z){float scale=(u_contact_floor.z/speed)*0.9999995231628418;tangent*=scale;normal*=scale;}
      // Use the actual post-force approach speed. A small upward velocity
      // can still be overcome by gravity before the sweep; exact vy==0 loses
      // floor traction for these supported substeps. Separating motion gets zero.
      // The sweep spent min(tangent,mu*normal) stopping existing sliding.
      // Only the unused static traction may resist later position corrections.
      b_packed[id*8u+6u]=vec4(max(0.0,mu*normal-tangent)*u_contact_floor.y,0.0,0.0,0.0);
    }
  }
  b_packed[id*8u+4u]=p;
  // enabled, material friction, validation error, reserved
  b_packed[id*8u+5u]=vec4(1.0,mu,valid?0.0:1.0,0.0);
}
