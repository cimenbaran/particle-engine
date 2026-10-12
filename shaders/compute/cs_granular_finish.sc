#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "granular_io.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_velocity.x))return;
  uint base=id*4u;for(uint lane=0u;lane<4u;++lane)b_candidate[base+lane]=b_input[base+lane];
  if(b_upstream[0]!=0u)return;
  uint kind=granularSlot(id);if(kind==0u || kind==2u)return;if(kind==3u){atomicOr(b_status[0],1u);return;}
  vec3 p=b_input[base].xyz,lo=vec3(-7.9375,0.0625,-7.9375),hi=vec3(7.9375,15.9375,7.9375);
  if(any(lessThan(p,lo)) || any(greaterThan(p,hi))){atomicOr(b_status[0],1u);return;}
  vec4 motion=b_input[base+1u];vec3 v=motion.xyz,old=v;
  // New outward velocity introduced by contacts is removed inelastically.
  bool x=(p.x<=lo.x && v.x<0.0)||(p.x>=hi.x && v.x>0.0);
  bool y=(p.y<=lo.y && v.y<0.0)||(p.y>=hi.y && v.y>0.0);
  bool z=(p.z<=lo.z && v.z<0.0)||(p.z>=hi.z && v.z>0.0);
  if(x)v.x=0.0;if(y)v.y=0.0;if(z)v.z=0.0;
  if(x || y || z) {
    uint material=floatBitsToUint(b_input[base+3u]).x;float mu=b_materials[material*4u].y;
    vec3 impulse=v-old;float speed=sqrt(dot(v,v)),support=sqrt(dot(impulse,impulse));
    if(speed>0.0)v*=max(0.0,1.0-mu*support/speed);
    atomicAdd(b_status[2],1u);
  }
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  if(largest>0.0) {
    vec3 scaled=largest>18446744073709551616.0?v*5.421010862427522e-20:v;
    float magnitude=max(abs(scaled.x),max(abs(scaled.y),abs(scaled.z)));
    vec3 unit=scaled/magnitude;float length=sqrt(dot(unit,unit));
    if(largest>u_granular.y/length){v=unit*((u_granular.y/length)*0.9999995231628418);atomicAdd(b_status[1],1u);}
  }
  if(!contactFinite3(v)){atomicOr(b_status[0],1u);return;}
  motion.xyz=v;b_candidate[base+1u]=motion;
}
