#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "materials.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_history,vec4,1);
BUFFER_RO(b_tracking,uint,2);
BUFFER_RW(b_instances,vec4,4);
uniform vec4 u_particle_debug; // capacity, materials, mode, speed scale
bool finiteValue(float v){return (floatBitsToUint(v)&0x7f800000u)!=0x7f800000u;}
int signedDebugCell(float v) {
  uint bits=floatBitsToUint(v);
  if((bits&0x7f800000u)==0u)return (bits&0x80000000u)!=0u && (bits&0x007fffffu)!=0u?-1:0;
  return int(floor(v*8.0));
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_particle_debug.x))return;
  vec4 p=b_state[id*4u];uvec4 identity=floatBitsToUint(b_state[id*4u+3u]);
  b_instances[id*2u]=vec4_splat(0.0);b_instances[id*2u+1u]=vec4_splat(0.0);
  if((identity.y&1u)==0u || identity.x>=uint(u_particle_debug.y) ||
     !finiteValue(p.x) || !finiteValue(p.y) || !finiteValue(p.z) || !finiteValue(p.w) || p.w<=0.0)return;
  uint color=materialColor(identity.x),mode=uint(u_particle_debug.z);
  if(mode==1u) {
    vec3 v=b_state[id*4u+1u].xyz;
    if(!finiteValue(v.x) || !finiteValue(v.y) || !finiteValue(v.z))color=0xffff00ffu;
    else {
      float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
      float t=largest>=u_particle_debug.w?1.0:min(length(v)/u_particle_debug.w,1.0);
      uvec3 c=uvec3(vec3(32.0+223.0*t,96.0*(1.0-t),255.0*(1.0-t))+vec3_splat(0.5));
      color=c.x|(c.y<<8u)|(c.z<<16u)|0xff000000u;
    }
  } else if(mode==4u) {
    vec3 v=b_state[id*4u+1u].xyz;
    if(!finiteValue(v.x) || !finiteValue(v.y) || !finiteValue(v.z))color=0xffff00ffu;
    else {
      // Construct exponent-scaled components without dividing by an extreme
      // float: reciprocal(max-float) can flush to zero on GPU backends.
      uvec3 bits=floatBitsToUint(v),exponents=(bits>>23u)&uvec3_splat(255u);
      uint largest=max(exponents.x,max(exponents.y,exponents.z));
      vec3 scaled=vec3_splat(0.0);
      for(int axis=0;axis<3;++axis) {
        if(exponents[axis]!=0u && largest-exponents[axis]<127u) {
          float mantissa=uintBitsToFloat((bits[axis]&0x807fffffu)|0x3f800000u);
          float factor=uintBitsToFloat((127u-(largest-exponents[axis]))<<23u);
          scaled[axis]=mantissa*factor;
        }
      }
      vec3 direction=vec3_splat(0.0);
      if(largest!=0u)direction=scaled/sqrt(dot(scaled,scaled));
      uvec3 c=uvec3((direction*0.5+vec3_splat(0.5))*255.0+vec3_splat(0.5));
      color=c.x|(c.y<<8u)|(c.z<<16u)|0xff000000u;
    }
  } else if(mode==2u) {
    if(p.x< -8.0 || p.x>=8.0 || p.y<0.0 || p.y>=16.0 || p.z< -8.0 || p.z>=8.0)color=0xffff00ffu;
    else {
      uint x=uint(signedDebugCell(p.x)+64),y=uint(signedDebugCell(p.y)),z=uint(signedDebugCell(p.z)+64);
      uint key=x+128u*(y+128u*z),hash=key*1664525u+1013904223u;
      color=(64u+(hash&127u))|((64u+((hash>>8u)&127u))<<8u)|((64u+((hash>>16u)&127u))<<16u)|0xff000000u;
    }
  } else if(mode==3u) {
    uvec4 h=floatBitsToUint(b_history[id]);color=0xff808080u;
    if(b_tracking[0]!=0u)color=0xffff00ffu;
    else if(h.x==identity.w && identity.w!=0u) {
      if(h.w==1u)color=h.z==0u?0xff60dc40u:0xff2080ffu;
      else if(h.y!=0u)color=0xff20c8ffu;
    }
  }
  b_instances[id*2u]=p;
  b_instances[id*2u+1u]=uintBitsToFloat(uvec4(color,floatBitsToUint(b_materials[identity.x*4u+3u]).z,identity.z,identity.y));
}
