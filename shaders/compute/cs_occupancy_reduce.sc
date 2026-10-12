#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_ranges,uint,0);
BUFFER_RO(b_meta,uint,1);
BUFFER_RO(b_upstream,uint,2);
BUFFER_RW(b_status,uint,3);
uniform vec4 u_occupancy;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){
  // Validate immutable inputs, never read a concurrently updated error word.
  uint cells=b_meta[0],records=b_meta[1],id=gl_GlobalInvocationID.x;
  if(b_meta[2]!=0u || b_upstream[0]!=0u || cells>records || records>uint(u_occupancy.x) || id>=cells)return;
  uint key=b_ranges[id*3u],begin=b_ranges[id*3u+1u],end=b_ranges[id*3u+2u];
  if(key>=2097152u || begin>=end || end>records){atomicOr(b_status[0],4u);return;}
  atomicMax(b_status[3],end-begin);
}
