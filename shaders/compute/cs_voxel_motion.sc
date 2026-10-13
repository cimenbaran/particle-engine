#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "granular_io.sh"
BUFFER_RO(b_world,uvec4,5);
#include "voxel_world_contact.sh"
uniform vec4 u_voxel_motion; // speed limit, bounce threshold, event budget
bool voxelCoefficient(float x){return contactFinite(x) && x>=0.0 && x<=1.0;}
float voxelWallTime(float p,float v,float lo,float hi) {
  if(v==0.0)return 2.0;
  return clamp(((v>0.0?hi:lo)-p)/v,0.0,2.0);
}
#include "voxel_motion_math.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x))return;
  if(b_upstream[0]!=0u)return;
  uint base=id*4u;for(uint lane=0u;lane<4u;++lane)b_candidate[base+lane]=b_input[base+lane];
  uint slotKind=granularSlot(id);if(slotKind==0u || slotKind==2u)return;
  if(slotKind==3u){atomicOr(b_status[0],2u);return;}
  uint particleMaterial=floatBitsToUint(b_input[base+3u]).x;
  vec4 mechanical=b_materials[particleMaterial*4u];
  if(!voxelCoefficient(mechanical.y) || !voxelCoefficient(mechanical.z)){atomicOr(b_status[0],2u);return;}
  vec3 lo=vec3(ivec3(b_world[0].xyz))*.125+vec3_splat(.0625);
  vec3 hi=vec3(ivec3(b_world[0].xyz))*.125+vec3(b_world[1].xyz)*4.0-vec3_splat(.0625);
  vec3 p=b_input[base].xyz,v=b_input[base+1u].xyz;
  if(any(lessThan(p,lo)) || any(greaterThan(p,hi))){atomicOr(b_status[0],2u);return;}
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  if(largest>0.0) {
    vec3 scaled=largest>18446744073709551616.0?v*5.421010862427522e-20:v;
    float magnitude=max(abs(scaled.x),max(abs(scaled.y),abs(scaled.z)));
    vec3 unit=scaled/magnitude;float speed=sqrt(dot(unit,unit));
    if(largest>u_voxel_motion.x/speed){v=unit*((u_voxel_motion.x/speed)*.9999995231628418);atomicAdd(b_status[1],1u);}
  }
  float remaining=u_velocity.y;uint events=0u,normals=0u;uint contactCount=0u,excluded=0xffffffffu;
  vec3 safeStart=p;
  LOOP while(remaining>0.0) {
    safeStart=p;
    vec3 wallTimes=vec3(voxelWallTime(p.x,v.x,lo.x,hi.x),voxelWallTime(p.y,v.y,lo.y,hi.y),voxelWallTime(p.z,v.z,lo.z,hi.z));
    float wall=min(wallTimes.x,min(wallTimes.y,wallTimes.z)),duration=min(remaining,wall);
    vec3 end=clamp(p+v*duration,lo,hi),centre,normal;float time;uint material,index;
    uint kind=voxelWorldSweep(p,end,true,excluded,centre,normal,time,material,index);
    if(kind>=2u){atomicOr(b_status[0],kind==2u?2u:8u);return;}
    if(kind==0u && wall>remaining){p=end;remaining=0.0;break;}
    if(events>=uint(u_voxel_motion.z)){atomicOr(b_status[0],4u);return;}
    ++events;
    if(kind==1u) {
      if(material>=uint(u_velocity.z)){atomicOr(b_status[0],2u);return;}
      vec4 terrainMaterial=b_materials[material*4u];
      if(floatBitsToUint(b_materials[material*4u+3u]).y!=0u || !voxelCoefficient(terrainMaterial.y) || !voxelCoefficient(terrainMaterial.z)){atomicOr(b_status[0],2u);return;}
      float incoming=dot(v,normal),outgoing=-incoming>u_voxel_motion.y?-min(mechanical.z,terrainMaterial.z)*incoming:0.0;
      vec3 tangent=v-normal*incoming;float speed=length(tangent),support=max(0.0,outgoing-incoming);
      float scale=speed>0.0?max(0.0,1.0-sqrt(mechanical.y*terrainMaterial.y)*support/speed):0.0;
      v=normal*outgoing+tangent*scale;vec3 separated;float fraction;
      if(!voxelMotionSeparate(p,centre,normal,separated,fraction)){atomicOr(b_status[0],8u);return;}p=separated;
      // Gather the complete local shell, including nearby contacts not yet
      // hit in this substep. Otherwise rounded endpoints can alternate between
      // two tangent contacts forever while missing the opposite gap surfaces.
      contactCount=0u;vec3 contactNormals[8];
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
        if(contactCount==8u){atomicOr(b_status[0],4u);return;}
        contactNormals[contactCount]=voxelNormal(centre,lower);++contactCount;
      }
      if(contactCount>1u)v=voxelContactCone(v,contactNormals,contactCount);
      // This convex voxel cannot be entered again by the outgoing straight
      // ray. Re-enable it only after another collider changes the velocity.
      excluded=index;remaining=max(0.0,remaining-duration*time*fraction);++normals;
    } else {
      excluded=0xffffffffu;p=end;remaining=max(0.0,remaining-wall);vec3 old=v;
      bool x=wallTimes.x==wall,y=wallTimes.y==wall,z=wallTimes.z==wall;
      if(x){v.x=abs(old.x)>u_voxel_motion.y?-mechanical.z*old.x:0.0;++normals;}
      if(y){v.y=abs(old.y)>u_voxel_motion.y?-mechanical.z*old.y:0.0;++normals;}
      if(z){v.z=abs(old.z)>u_voxel_motion.y?-mechanical.z*old.z:0.0;++normals;}
      vec3 tangent=vec3(x?0.0:v.x,y?0.0:v.y,z?0.0:v.z);float speed=length(tangent),support=length(v-old);
      if(speed>0.0){float scale=max(0.0,1.0-mechanical.y*support/speed);if(!x)v.x*=scale;if(!y)v.y*=scale;if(!z)v.z*=scale;}
    }
    if(!contactFinite3(p) || !contactFinite3(v)){atomicOr(b_status[0],8u);return;}
  }
  // A final stored endpoint must be outside occupancy. Tiny float penetration
  // is repaired geometrically without creating a collision or velocity impulse.
  LOOP for(uint repair=0u;repair<4u;++repair) {
    vec3 centre,normal;float time;uint material,index;
    uint kind=voxelWorldSweep(p,p,false,0xffffffffu,centre,normal,time,material,index);
    if(kind==0u || kind==1u)break;
    if(kind!=2u){atomicOr(b_status[0],8u);return;}
    vec3 lower=voxelMotionLower(index);
    float distance=length(p-clamp(p,lower,lower+vec3_splat(.125)));
    if(distance<.0625-2e-6 || repair==3u){atomicOr(b_status[0],8u);return;}
    vec3 separated;float fraction;
    if(!voxelMotionSeparate(safeStart,p+normal*(.0625-distance),normal,separated,fraction)){atomicOr(b_status[0],8u);return;}p=separated;
  }
  if(!contactFinite3(p) || !contactFinite3(v) || any(lessThan(p,lo)) || any(greaterThan(p,hi))){atomicOr(b_status[0],8u);return;}
  vec4 pose=b_input[base],motion=b_input[base+1u];pose.xyz=p;motion.xyz=v;b_candidate[base]=pose;b_candidate[base+1u]=motion;
  atomicAdd(b_status[2],events);atomicAdd(b_status[3],normals);
}
