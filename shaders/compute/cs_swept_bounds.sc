#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_input,vec4,0);
BUFFER_RO(b_materials,vec4,1);
BUFFER_RO(b_upstream,uint,2);
BUFFER_RW(b_candidate,vec4,3);
BUFFER_RW(b_status,uint,4);
uniform vec4 u_velocity; // capacity, dt, material count
uniform vec4 u_sweep; // speed limit, bounce threshold, event budget
uniform vec4 u_sweep_sphere; // fixed sphere centre/radius; zero radius disables
#include "sphere_contact.sh"
bool coefficient(float v) {uint b=floatBitsToUint(v);return (b&0x7fffffffu)==0u || b<=0x3f800000u;}
float hitTime(float p,float v,float lo,float hi) {
  if(v==0.0) return 2.0;
  float distance=(v>0.0?hi:lo)-p;
  if(distance==0.0) return 0.0;
  return clamp(distance/v,0.0,2.0);
}
float reflected(float v,float e) {return abs(v)>u_sweep.y?-e*v:0.0;}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x)) return;
  uint base=id*4u;
  for(uint lane=0u;lane<4u;++lane) b_candidate[base+lane]=b_input[base+lane];
  if(b_upstream[0]!=0u) return;
  uvec4 identity=floatBitsToUint(b_input[base+3u]);if((identity.y&1u)==0u) return;
  if(identity.x>=uint(u_velocity.z)) {atomicOr(b_status[0],2u);return;}
  uint phase=floatBitsToUint(b_materials[identity.x*4u+3u]).y;
  if(phase>2u) {atomicOr(b_status[0],2u);return;}if(phase!=0u) return;
  vec4 pose=b_input[base],motion=b_input[base+1u],material=b_materials[identity.x*4u];
  if(identity.z!=id || identity.w==0u || pose.w!=0.0625 || !validMass(motion.w) ||
     !contactFinite3(pose.xyz) || !contactFinite3(motion.xyz) || !coefficient(material.y) || !coefficient(material.z)) {
    atomicOr(b_status[0],2u);return;
  }
  if(motion.w==0.0) return;
  vec3 lo=vec3(-7.9375,0.0625,-7.9375),hi=vec3(7.9375,15.9375,7.9375);
  if(any(lessThan(pose.xyz,lo)) || any(greaterThan(pose.xyz,hi))) {atomicOr(b_status[0],2u);return;}
  vec3 p=pose.xyz,v=motion.xyz;
  if(u_sweep_sphere.w>0.0 && length(p-u_sweep_sphere.xyz)<u_sweep_sphere.w+pose.w){atomicOr(b_status[0],2u);return;}
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  if(largest>0.0) {
    vec3 scaled=largest>18446744073709551616.0?v*5.421010862427522e-20:v;
    float magnitude=max(abs(scaled.x),max(abs(scaled.y),abs(scaled.z)));
    vec3 unit=scaled/magnitude;float length=sqrt(dot(unit,unit));
    if(largest>u_sweep.x/length) {
      // Margin keeps float rounding from publishing a speed above the cap.
      v=unit*((u_sweep.x/length)*0.9999995231628418);atomicAdd(b_status[1],1u);
    }
  }
  float remaining=u_velocity.y;uint events=0u,planes=0u;bool sphereEligible=true;
  LOOP while(remaining>0.0) {
    vec3 times=vec3(hitTime(p.x,v.x,lo.x,hi.x),hitTime(p.y,v.y,lo.y,hi.y),hitTime(p.z,v.z,lo.z,hi.z));
    vec4 sphere=vec4(0.0,0.0,0.0,2.0);
    if(sphereEligible)sphere=sphereImpact(p,v,u_sweep_sphere);
    // Division can round an endpoint root just beyond dt. A contained endpoint
    // independently proves an entry occurred; retain that collision at dt.
    if(sphere.w>remaining && sphere.w<2.0 && u_sweep_sphere.w>0.0 &&
       length(p+v*remaining-u_sweep_sphere.xyz)<=u_sweep_sphere.w+pose.w)sphere.w=remaining;
    float wall=min(times.x,min(times.y,times.z)),first=min(wall,sphere.w),travel=min(first,remaining);
    p+=v*travel;
    if(first>remaining) break;
    if(events>=uint(u_sweep.z)) {atomicOr(b_status[0],4u);return;}
    ++events;remaining=max(0.0,remaining-first);
    if(sphere.w<wall) {
      vec3 normal=sphere.xyz;float incoming=dot(v,normal);
      float outgoing=-incoming>u_sweep.y?-material.z*incoming:0.0;
      vec3 tangent=v-normal*incoming;float speed=length(tangent),support=max(0.0,outgoing-incoming);
      float scale=speed>0.0?max(0.0,1.0-material.y*support/speed):0.0;
      v=normal*outgoing+tangent*scale;
      // Keep the published contact point just outside the rounded surface.
      // Event eligibility below separately excludes impossible repeat impacts.
      p=u_sweep_sphere.xyz+normal*(u_sweep_sphere.w+pose.w+sphereSkin);++planes;
      // An outgoing straight ray cannot re-enter this fixed convex sphere.
      // Only a subsequent wall response can turn it back. Re-testing a tangent
      // ray here can invent another impact from rounded distance estimates.
      sphereEligible=false;
      continue;
    }
    sphereEligible=true;
    bool x=times.x==first,y=times.y==first,z=times.z==first;
    vec3 old=v;
    if(x) {p.x=old.x>0.0?hi.x:lo.x;v.x=reflected(old.x,material.z);++planes;}
    if(y) {p.y=old.y>0.0?hi.y:lo.y;v.y=reflected(old.y,material.z);++planes;}
    if(z) {p.z=old.z>0.0?hi.z:lo.z;v.z=reflected(old.z,material.z);++planes;}
    vec3 tangent=vec3(x?0.0:v.x,y?0.0:v.y,z?0.0:v.z),impulse=v-old;
    float speed=sqrt(dot(tangent,tangent)),support=sqrt(dot(impulse,impulse));
    if(speed>0.0) {
      float scale=max(0.0,1.0-material.y*support/speed);
      if(!x) v.x*=scale;if(!y) v.y*=scale;if(!z) v.z*=scale;
    }
  }
  if(!contactFinite3(p) || !contactFinite3(v)) {atomicOr(b_status[0],8u);return;}
  // Remove only endpoint rounding outside the planes after an otherwise valid sweep.
  pose.xyz=clamp(p,lo,hi);motion.xyz=v;b_candidate[base]=pose;b_candidate[base+1u]=motion;
  atomicAdd(b_status[2],events);atomicAdd(b_status[3],planes);
}
