#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "granular_grid.sh"
#include "contact_math.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RW(b_corrections,vec4,1);
BUFFER_RO(b_world,uvec4,2);
BUFFER_RO(b_materials,vec4,3);
#include "voxel_world_contact.sh"
#include "voxel_motion_math.sh"
uniform vec4 u_contacts;
uniform vec4 u_contact_bounds;
uniform vec4 u_contact_friction;
// Verify the whole separation leg too: clear endpoints alone do not prove a
// chord through multiple rounded voxel boundaries is collision-free.
bool voxelCorrectionSeparate(vec3 start,vec3 contact,vec3 normal,out vec3 proposed,out float fraction) {
  if(!voxelMotionSeparate(start,contact,normal,proposed,fraction))return false;
  vec3 centre,n;float time;uint material,index;
  uint kind=voxelWorldSweep(start,proposed,true,0xffffffffu,centre,n,time,material,index);
  if(kind==0u || (kind==1u && time==1.0))return true;
  proposed=start;fraction=0.0;return voxelMotionClear(start);
}
// Constrain a displacement, not a velocity: no restitution or second kinetic
// friction impulse. Each leg is swept. Path length pays for separation skins,
// so no final straight-line rescaling can cut through nonconvex occupancy.
uint voxelCorrection(vec3 original,vec3 delta,out vec3 result) {
  result=original;
  vec3 lo=vec3(ivec3(b_world[0].xyz))*.125+vec3_splat(.0625);
  vec3 hi=vec3(ivec3(b_world[0].xyz))*.125+vec3(b_world[1].xyz)*4.0-vec3_splat(.0625);
  if(u_contact_bounds.x!=0.0) {
    lo=max(lo,vec3(-PW_GRID_EXTENT,0.0,-PW_GRID_EXTENT)+vec3_splat(.0625));
    hi=min(hi,vec3(PW_GRID_EXTENT,16.0,PW_GRID_EXTENT)-vec3_splat(.0625));
  }
  if(any(lessThan(original,lo)) || any(greaterThan(original,hi)) || !voxelMotionClear(original))return 6u;
  vec3 p=original,remaining=delta;float budget=u_contacts.w;
  uint contacts[8];uint contactCount=0u,excluded=0xffffffffu;
  LOOP for(uint event=0u;event<16u;++event) {
    float distance=length(remaining);
    if(distance>budget && distance>0.0)remaining*=budget/distance;
    vec3 end=clamp(p+remaining,lo,hi),centre,normal;float time;uint material,index;
    remaining=end-p;
    uint kind=voxelWorldSweep(p,end,true,excluded,centre,normal,time,material,index);
    if(kind>=2u)return kind==2u?6u:5u;
    if(kind==0u) {
      if(!voxelMotionClear(end)) {
        // A closed endpoint can round microscopically inward. Keep the last
        // known clear point; never accept an unchecked penetrating endpoint.
        vec3 c,n;float t;uint m,i;
        uint endpoint=voxelWorldSweep(end,end,false,0xffffffffu,c,n,t,m,i);
        if(endpoint!=2u)return 5u;
        vec3 lower=voxelMotionLower(i);float gap=length(end-clamp(end,lower,lower+vec3_splat(.125)));
        if(gap<.0625-2e-6)return 6u;
        vec3 separated;float fraction;
        if(!voxelCorrectionSeparate(p,end+n*(.0625-gap),n,separated,fraction))return 5u;
        if(length(separated-p)<=budget && all(greaterThanEqual(separated,lo)) && all(lessThanEqual(separated,hi)))end=separated;
        else end=p;
      }
      result=end;return 0u;
    }
    if(material>=uint(u_contact_friction.y) || floatBitsToUint(b_materials[material*4u+3u]).y!=0u)return 4u;
    vec3 separated;float fraction;
    if(!voxelCorrectionSeparate(p,centre,normal,separated,fraction))return 5u;
    float travelled=length(separated-p);
    if(travelled>budget || any(lessThan(separated,lo)) || any(greaterThan(separated,hi))) {
      // Skin cannot spend more than the available displacement. Retreat on
      // the swept incoming segment instead; its start is always admissible.
      separated=p;fraction=0.0;travelled=0.0;
    }
    remaining*=max(0.0,1.0-time*fraction);
    remaining-=normal*min(0.0,dot(remaining,normal));
    uint kept=0u;bool present=false;vec3 normals[8];
    LOOP for(uint c=0u;c<contactCount;++c) {
      uint address=contacts[c];vec3 lower=voxelMotionLower(address);
      if(length(centre-clamp(centre,lower,lower+vec3_splat(.125)))>.0625+4e-6)continue;
      contacts[kept]=address;normals[kept]=voxelNormal(centre,lower);++kept;if(address==index)present=true;
    }
    contactCount=kept;
    if(!present) {
      if(contactCount==8u)return 8u;
      contacts[contactCount]=index;normals[contactCount]=voxelNormal(centre,voxelMotionLower(index));++contactCount;
    }
    // Revisiting a retained contact means the two-surface projection did not
    // clear this local gap. Include unvisited nearby surfaces before retrying.
    if(present) {
    // Gather the entire local shell. Retaining only previously hit voxels
    // can alternate between tangent endpoints without ever seeing the other
    // surfaces that constrain this correction in a narrow rounded gap.
    contactCount=0u;
    ivec3 minimum=ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
    ivec3 first=max(ivec3(ceil((centre-vec3_splat(.062504))*8.0))-ivec3(1,1,1)-minimum,ivec3(0,0,0));
    ivec3 last=min(ivec3(floor((centre+vec3_splat(.062504))*8.0))-minimum,dimensions*32-ivec3(1,1,1));
    LOOP for(int z=first.z;z<=last.z;++z)
    LOOP for(int y=first.y;y<=last.y;++y)
    LOOP for(int x=first.x;x<=last.x;++x) {
      ivec3 cell=ivec3(x,y,z),chunk=cell/32,local=cell-chunk*32;
      uint address=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z))*32768u+uint(local.x+32*(local.y+32*local.z));
      uvec4 packed=b_world[4u+address/4u];if(packed[address%4u]==0xffffffffu)continue;
      vec3 lower=vec3(minimum+cell)*.125;
      if(length(centre-clamp(centre,lower,lower+vec3_splat(.125)))>.062504)continue;
      if(contactCount==8u)return 8u;
      contacts[contactCount]=address;normals[contactCount]=voxelNormal(centre,lower);++contactCount;
    }
    }
    if(contactCount>1u)remaining=voxelContactCone(remaining,normals,contactCount);
    budget=max(0.0,budget-travelled);p=separated;excluded=index;
    if(!contactFinite3(p) || !contactFinite3(remaining))return 5u;
    if(budget==0.0 || dot(remaining,remaining)==0.0){result=p;return 0u;}
  }
  return 8u;
}
// Consider each contact's own Coulomb budget; do not sum duplicated voxel
// faces. Keep one admissible sticking correction, then sweep it again.
void voxelTractionCandidate(vec3 position,vec3 requested,vec3 baseline,vec3 normal,
    float mu,vec4 forced,bool supported,inout vec3 best) {
  vec3 relative=position-baseline;
  vec3 tangent=relative-normal*dot(relative,normal);
  float budget=mu*max(0.0,dot(position-requested,normal));
  if(supported && forced.w>0.0) {
    float normalSpeed=max(0.0,-dot(forced.xyz,normal));
    vec3 sliding=forced.xyz-normal*dot(forced.xyz,normal);
    // Prediction already spent friction on pre-existing tangent velocity.
    budget+=max(0.0,mu*normalSpeed-length(sliding))*forced.w;
  }
  float magnitude=length(tangent);
  if(magnitude<=budget && magnitude>length(best))best=tangent;
}
uint voxelTraction(uint slot,vec3 original,vec3 requested,inout vec3 proposed) {
  if(b_state[slot*8u+5u].x==0.0)return 0u;
  float own=b_state[slot*8u+5u].y;
  vec3 baseline=b_state[slot*8u+4u].xyz,best=vec3_splat(0.0);
  vec4 forced=b_state[slot*8u+7u];
  ivec3 minimum=ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
  vec3 lo=vec3(minimum)*.125+vec3_splat(.0625);
  vec3 hi=vec3(minimum)*.125+vec3(dimensions)*4.0-vec3_splat(.0625);
  LOOP for(uint axis=0u;axis<3u;++axis) {
    vec3 n=vec3_splat(0.0);bool supported=false;
    if(proposed[axis]<=lo[axis]+4e-6){n[axis]=1.0;supported=baseline[axis]<=lo[axis]+4e-6;}
    else if(proposed[axis]>=hi[axis]-4e-6){n[axis]=-1.0;supported=baseline[axis]>=hi[axis]-4e-6;}
    if(n[axis]!=0.0)voxelTractionCandidate(proposed,requested,baseline,n,own,forced,supported,best);
  }
  ivec3 first=max(ivec3(ceil((proposed-vec3_splat(.062504))*8.0))-ivec3(1,1,1)-minimum,ivec3(0,0,0));
  ivec3 last=min(ivec3(floor((proposed+vec3_splat(.062504))*8.0))-minimum,dimensions*32-ivec3(1,1,1));
  LOOP for(int z=first.z;z<=last.z;++z)
  LOOP for(int y=first.y;y<=last.y;++y)
  LOOP for(int x=first.x;x<=last.x;++x) {
    ivec3 cell=ivec3(x,y,z),chunk=cell/32,local=cell-chunk*32;
    uint address=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z))*32768u+uint(local.x+32*(local.y+32*local.z));
    uvec4 packed=b_world[4u+address/4u];uint material=packed[address%4u];if(material==0xffffffffu)continue;
    vec3 lower=vec3(minimum+cell)*.125;
    if(length(proposed-clamp(proposed,lower,lower+vec3_splat(.125)))>.062504)continue;
    if(material>=uint(u_contact_friction.y) || floatBitsToUint(b_materials[material*4u+3u]).y!=0u)return 4u;
    float other=b_materials[material*4u].y;uint bits=floatBitsToUint(other);
    if((bits&0x7fffffffu)>0x3f800000u || ((bits&0x80000000u)!=0u && (bits&0x7fffffffu)!=0u))return 4u;
    bool supported=length(baseline-clamp(baseline,lower,lower+vec3_splat(.125)))<=.062504;
    voxelTractionCandidate(proposed,requested,baseline,voxelNormal(proposed,lower),sqrt(own*other),forced,supported,best);
  }
  if(dot(best,best)>0.0) {
    vec3 result;uint error=voxelCorrection(original,proposed-best-original,result);
    if(error!=0u)return error;proposed=result;
  }
  return 0u;
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_contacts.x))return;
  uvec4 counts=floatBitsToUint(b_corrections[slot*2u+1u]);
  if(counts.w!=0u || b_state[slot*8u+1u].w==0.0)return;
  vec3 original=b_state[slot*8u].xyz,proposed;
  if(!voxelMotionClear(original))counts.w=6u;
  else if(b_state[slot*8u+5u].x!=0.0 && !voxelMotionClear(b_state[slot*8u+4u].xyz))counts.w=7u;
  else counts.w=voxelCorrection(original,b_corrections[slot*2u].xyz,proposed);
  if(counts.w==0u)counts.w=voxelTraction(slot,original,original+b_corrections[slot*2u].xyz,proposed);
  if(counts.w==0u) {
    vec3 delta=proposed-original,stored=original+delta;
    if(!contactFinite3(stored) || length(delta)>u_contacts.w+2e-6 || !voxelMotionClear(stored))counts.w=5u;
    else if(stored.x< -PW_GRID_EXTENT || stored.x>=PW_GRID_EXTENT || stored.y<0.0 || stored.y>=16.0 || stored.z< -PW_GRID_EXTENT || stored.z>=PW_GRID_EXTENT)counts.w=6u;
    else b_corrections[slot*2u]=vec4(delta,0.0);
  }
  b_corrections[slot*2u+1u]=uintBitsToFloat(counts);
}
