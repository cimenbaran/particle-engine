#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_source,vec4,0);
BUFFER_RO(b_history,uvec4,1);
BUFFER_RO(b_materials,uvec4,2); // inverse-mass bits, solid, pair admissible, unused
BUFFER_RO(b_world,uvec4,3);
BUFFER_RW(b_candidates,uvec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_deposit; // capacity, material count, unused, unused
int depositCell(float value){uint bits=floatBitsToUint(value);if((bits&0x7fffffffu)!=0u && (bits&0x7fffffffu)<0x00800000u)return (bits&0x80000000u)!=0u?-1:0;return int(floor(value*8.0));}
bool depositVoxel(ivec3 cell,out uint material){
 ivec3 relative=cell-ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
 if(any(lessThan(relative,ivec3(0,0,0))) || any(greaterThanEqual(relative,dimensions*32)))return false;
 ivec3 chunk=relative/32,local=relative-chunk*32;
 uint slot=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z));
 uint address=slot*32768u+uint(local.x+32*(local.y+32*local.z));
 material=b_world[4u+address/4u][address%4u];return true;
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){
 uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_deposit.x))return;
 b_candidates[slot*2u]=uvec4(0u,0u,0u,0u);b_candidates[slot*2u+1u]=uvec4(slot,0u,0u,0u);
 uvec4 id=floatBitsToUint(b_source[slot*4u+3u]);if(id.y==0u)return;
 if(id.y!=1u || id.z!=slot || id.w==0u || id.x>=uint(u_deposit.y)){atomicOr(b_status[0],1u);return;}
 uvec4 material=b_materials[id.x];if(material.y==0u)return;
 vec4 p=b_source[slot*4u],v=b_source[slot*4u+1u];
 uint radiusBits=floatBitsToUint(p.w),massBits=floatBitsToUint(v.w);
 if(!contactFinite3(p.xyz) || !contactFinite3(v.xyz) || !contactFinite(p.w) || !contactFinite(v.w) ||
    (radiusBits&0x80000000u)!=0u || (radiusBits&0x7fffffffu)==0u || p.w>1.0 ||
    ((massBits&0x80000000u)!=0u && (massBits&0x7fffffffu)!=0u) || any(greaterThan(abs(p.xyz),vec3_splat(1e6)))){atomicOr(b_status[0],2u);return;}
 // Solid radius subnormals cannot safely participate in float clearance on Metal.
 if((radiusBits&0x7fffffffu)<0x00800000u){atomicOr(b_status[0],2u);return;}
 uint ignored;atomicFetchAndAdd(b_status[3],1u,ignored);
 uvec4 h=b_history[slot];float speedSquared=dot(v.xyz,v.xyz);
 // Rounded-up bound defers only the narrow threshold band. It cannot admit a
 // grain moving above the CPU hold limit due to binary32 dot-product rounding.
 bool eligible=p.w==0.0625 && massBits==material.x && material.z!=0u &&
  h.x==id.w && h.y==30u && h.z==0u && h.w==1u && speedSquared*1.000001<=0.01;
 ivec3 cell=ivec3(depositCell(p.x),depositCell(p.y),depositCell(p.z));
 uint ready=0u;
 if(eligible){
  atomicFetchAndAdd(b_status[1],1u,ignored);
  if(cell.x>=-128 && cell.x<=127 && cell.y>=0 && cell.y<=127 && cell.z>=-128 && cell.z<=127){
   uint occupancy=0u;
   if(depositVoxel(cell,occupancy) && occupancy==0xffffffffu){
    if(cell.y==0)ready=1u;
    else{uint support=0u;if(depositVoxel(cell-ivec3(0,1,0),support) && support<uint(u_deposit.y) && b_materials[support].y!=0u)ready=1u;}
   }
  }
 }
 if(ready!=0u)atomicFetchAndAdd(b_status[2],1u,ignored);
 b_candidates[slot*2u]=uvec4(uint(cell.x),uint(cell.y),uint(cell.z),id.x);
 b_candidates[slot*2u+1u]=uvec4(slot,id.w,eligible?2u:1u,ready);
}
