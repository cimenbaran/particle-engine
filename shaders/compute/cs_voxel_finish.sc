#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "granular_io.sh"
BUFFER_RO(b_world,uvec4,5);
#include "voxel_world_contact.sh"
uniform vec4 u_voxel_motion;
bool finishCoefficient(float value){uint bits=floatBitsToUint(value),magnitude=bits&0x7fffffffu;return magnitude<=0x3f800000u && ((bits&0x80000000u)==0u || magnitude==0u);}
float finishSpeed(vec3 v) {
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  return largest>0.0?largest*length(v/largest):0.0;
}
// A .062504m shell spans at most 3 cells per axis (27 voxels), plus at
// most one resident plane per axis (3). All candidates fit; no truncation.
bool finishFeasible(vec3 v,vec3 normals[32],uint count,float tolerance) {
  LOOP for(uint i=0u;i<count;++i)if(dot(v,normals[i])< -tolerance)return false;
  return true;
}
// In 3D a nonzero projected velocity has at most two independent active
// normals. Track their nonnegative KKT impulses for material friction. Zero
// is always feasible and needs no friction. Ties retain first traversal order.
vec3 finishCone(vec3 v,vec3 normals[32],float friction[32],uint count,float scale) {
  if(finishFeasible(v,normals,count,5e-7*max(finishSpeed(v),.001*scale)))return v;
  // Maximize retained speed instead of subtracting nearly equal errors.
  // Otherwise a tiny free tangent disappears beside a large normal impulse.
  vec3 best=vec3_splat(0.0);float retained=0.0,budget=0.0;
  LOOP for(uint i=0u;i<count;++i) {
    float di=dot(v,normals[i]),ni2=dot(normals[i],normals[i]);
    // Cross form keeps roundoff proportional to the retained tangent rather
    // than the removed normal component, even for almost normal impacts.
    vec3 candidate=cross(normals[i],cross(v,normals[i]))/ni2;
    float speed=finishSpeed(candidate);
    if(di<=0.0 && speed>retained && finishFeasible(candidate,normals,count,5e-7*max(speed,.001*scale))) {
      best=candidate;retained=speed;budget=friction[i]*(-di/ni2);
    }
    LOOP for(uint j=0u;j<i;++j) {
      vec3 axis=cross(normals[i],normals[j]);float denominator=dot(axis,axis);
      if(denominator<=1e-12)continue;
      float cosine=dot(normals[i],normals[j]),dj=dot(v,normals[j]),nj2=dot(normals[j],normals[j]);
      float a=(-di*nj2+cosine*dj)/denominator,b=(-dj*ni2+cosine*di)/denominator;
      if(a<0.0 || b<0.0)continue;
      candidate=axis*(dot(v,axis)/denominator);speed=finishSpeed(candidate);
      if(speed>retained && finishFeasible(candidate,normals,count,5e-7*max(speed,.001*scale))) {
        best=candidate;retained=speed;budget=friction[i]*a+friction[j]*b;
      }
    }
  }
  float speed=finishSpeed(best);
  return speed>0.0?best*max(0.0,1.0-budget/speed):best;
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x))return;
  if(b_upstream[0]!=0u)return;
  uint base=id*4u;for(uint lane=0u;lane<4u;++lane)b_candidate[base+lane]=b_input[base+lane];
  uint kind=granularSlot(id);if(kind==0u || kind==2u)return;
  if(kind==3u){atomicOr(b_status[0],2u);return;}
  uint material=floatBitsToUint(b_input[base+3u]).x;float own=b_materials[material*4u].y;
  if(!finishCoefficient(own)){atomicOr(b_status[0],2u);return;}
  ivec3 minimum=ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
  vec3 p=b_input[base].xyz,lo=vec3(minimum)*.125+vec3_splat(.0625);
  vec3 hi=vec3(minimum)*.125+vec3(dimensions)*4.0-vec3_splat(.0625);
  if(any(lessThan(p,lo)) || any(greaterThan(p,hi))){atomicOr(b_status[0],2u);return;}
  vec3 v=b_input[base+1u].xyz;
  // Project before capping: a large normal impulse must not erase a valid
  // free tangent. Power-of-two preconditioning keeps dot products finite.
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  float scale=largest>1152921504606846976.0?3.3881317890172014e-21:1.0; // 2^-68
  v*=scale;
  vec3 normals[32];float friction[32];uint count=0u;
  LOOP for(uint a=0u;a<3u;++a) {
    vec3 n=vec3_splat(0.0);
    if(p[a]<=lo[a]+4e-6)n[a]=1.0;else if(p[a]>=hi[a]-4e-6)n[a]=-1.0;
    if(n[a]!=0.0){normals[count]=n;friction[count]=own;++count;}
  }
  ivec3 first=max(ivec3(ceil((p-vec3_splat(.062504))*8.0))-ivec3(1,1,1)-minimum,ivec3(0,0,0));
  ivec3 last=min(ivec3(floor((p+vec3_splat(.062504))*8.0))-minimum,dimensions*32-ivec3(1,1,1));
  LOOP for(int z=first.z;z<=last.z;++z)
  LOOP for(int y=first.y;y<=last.y;++y)
  LOOP for(int x=first.x;x<=last.x;++x) {
    ivec3 cell=ivec3(x,y,z),chunk=cell/32,local=cell-chunk*32;
    uint address=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z))*32768u+uint(local.x+32*(local.y+32*local.z));
    uvec4 packed=b_world[4u+address/4u];uint found=packed[address%4u];if(found==0xffffffffu)continue;
    vec3 lower=vec3(minimum+cell)*.125;
    // Compensated separation preserves strict nonpenetration at rounded edges.
    if(voxelSeparation(p,lower)<0.0){atomicOr(b_status[0],2u);return;}
    if(length(p-clamp(p,lower,lower+vec3_splat(.125)))>.062504)continue;
    if(found>=uint(u_velocity.z) || floatBitsToUint(b_materials[found*4u+3u]).y!=0u || !finishCoefficient(b_materials[found*4u].y)) {atomicOr(b_status[0],2u);return;}
    if(count==32u){atomicOr(b_status[0],4u);return;}
    normals[count]=voxelNormal(p,lower);friction[count]=sqrt(own*b_materials[found*4u].y);++count;
  }
  vec3 result=finishCone(v,normals,friction,count,scale);
  float speed=finishSpeed(result);
  if(speed>u_voxel_motion.x*scale) {
    result=(result/speed)*(u_voxel_motion.x*.9999995231628418);atomicAdd(b_status[1],1u);
  } else result/=scale;
  if(!contactFinite3(result) || !finishFeasible(result,normals,count,1e-5) || finishSpeed(result)>u_voxel_motion.x+2e-5){atomicOr(b_status[0],8u);return;}
  if(any(notEqual(result,b_input[base+1u].xyz)))atomicAdd(b_status[2],1u);
  atomicAdd(b_status[3],count);
  vec4 motion=b_input[base+1u];motion.xyz=result;b_candidate[base+1u]=motion;
}
