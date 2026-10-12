#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_support,vec4,1);
BUFFER_RW(b_history,vec4,2);
BUFFER_RW(b_status,uint,3);
uniform vec4 u_settling; // capacity, entry speed, exit speed
uniform vec4 u_settling_ticks; // bit-preserving uint dwell/grace counts
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_settling.x))return;
  uint generation=floatBitsToUint(b_state[id*4u+3u]).w;
  uvec4 empty=uvec4(generation,0u,0u,0u),h=floatBitsToUint(b_history[id]);
  vec4 support=b_support[id];uvec4 ticks=floatBitsToUint(u_settling_ticks);
  if(b_status[0]!=0u || support.w==0.0 || generation==0u){b_history[id]=uintBitsToFloat(empty);return;}
  if(h.x!=generation || h.w>1u || h.y>ticks.x || h.z>=ticks.y ||
     (h.w!=0u && h.y!=ticks.x) || (h.w==0u && (h.z!=0u || h.y==ticks.x)))h=empty;
  // Speeds above the bounded profile reject before squaring; this handles any
  // finite velocity without intermediate overflow or subnormal speed thresholds.
  vec3 v=b_state[id*4u+1u].xyz;float maxComponent=max(abs(v.x),max(abs(v.y),abs(v.z)));
  bool slow=false,retained=false;
  if(maxComponent<=u_settling.z){float squared=dot(v,v);slow=squared<=u_settling.y*u_settling.y;retained=squared<=u_settling.z*u_settling.z;}
  if(h.w!=0u) {
    if(!retained)h=empty;
    else if(support.y!=0.0)h.z=0u;
    else if(h.z==ticks.y-1u)h=empty;
    else h.z+=1u;
  } else if(slow && support.x!=0.0){h.y+=1u;if(h.y==ticks.x)h.w=1u;}
  else h=empty;
  b_history[id]=uintBitsToFloat(h);
  if(h.w!=0u){atomicAdd(b_status[1],1u);if(h.z!=0u)atomicAdd(b_status[3],1u);}
  else if(h.y!=0u)atomicAdd(b_status[2],1u);
}
