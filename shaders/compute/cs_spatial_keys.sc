#include "granular_grid.sh"
#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_materials,vec4,1);
BUFFER_RW(b_records,uint,2);
BUFFER_RW(b_status,uint,3);
uniform vec4 u_keys; // slot capacity, material count, reserved
bool finiteCoordinate(float value) {return (floatBitsToUint(value)&0x7f800000u)!=0x7f800000u;}
int signedCell(float value) {
  uint bits=floatBitsToUint(value);
  // Preserve floor for signed subnormals even on arithmetic that flushes them
  // to zero. Origin is zero and width is exactly 1/8 for this initial profile.
  if((bits&0x7f800000u)==0u) return (bits&0x80000000u)!=0u && (bits&0x007fffffu)!=0u?-1:0;
  return int(floor(value*8.0));
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;
  if(slot>=uint(u_keys.x)) return;
  b_records[slot*2u]=0xffffffffu;b_records[slot*2u+1u]=slot;
  uvec4 identity=floatBitsToUint(b_state[slot*4u+3u]);
  if((identity.y&1u)==0u) {atomicAdd(b_status[1],1u);return;}
  if(identity.x>=uint(u_keys.y)) {atomicAdd(b_status[2],1u);return;}
  uint phase=floatBitsToUint(b_materials[identity.x*4u+3u]).y;
  if(phase>2u) {atomicAdd(b_status[2],1u);return;}
  if(phase!=0u) {atomicAdd(b_status[1],1u);return;}
  vec3 position=b_state[slot*4u].xyz;
  if(identity.z!=slot || identity.w==0u || !finiteCoordinate(position.x) ||
     !finiteCoordinate(position.y) || !finiteCoordinate(position.z)) {atomicAdd(b_status[2],1u);return;}
  // Check bounds before float-to-int conversion; large finite inputs must not
  // overflow multiplication or conversion. Upper faces are excluded.
  if(position.x< -PW_GRID_EXTENT || position.x>=PW_GRID_EXTENT || position.y<0.0 || position.y>=16.0 ||
     position.z< -PW_GRID_EXTENT || position.z>=PW_GRID_EXTENT) {atomicAdd(b_status[3],1u);return;}
  ivec3 cell=ivec3(signedCell(position.x)+PW_GRID_OFFSET,signedCell(position.y),signedCell(position.z)+PW_GRID_OFFSET);
  if(cell.x<0 || cell.y<0 || cell.z<0 || cell.x>=PW_GRID_X || cell.y>=PW_GRID_Y || cell.z>=PW_GRID_Z)
    {atomicAdd(b_status[3],1u);return;}
  b_records[slot*2u]=uint(cell.x)+uint(PW_GRID_X)*(uint(cell.y)+uint(PW_GRID_Y)*uint(cell.z));
  atomicAdd(b_status[0],1u);
}
