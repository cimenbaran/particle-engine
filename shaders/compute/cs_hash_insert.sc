#include "granular_grid.sh"
#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "cell_hash.sh"
BUFFER_RO(b_ranges,uint,0);
BUFFER_RO(b_source,uint,1);
BUFFER_RW(b_table,uint,2);
BUFFER_RW(b_meta,uint,3);
BUFFER_RW(b_metrics,uint,4);
uniform vec4 u_hash;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x,capacity=uint(u_hash.x);
  if(b_source[2]!=0u || b_source[0]>capacity || b_source[1]>capacity || id>=b_source[0] || id>=capacity) return;
  uint key=b_ranges[id*3u],begin=b_ranges[id*3u+1u],end=b_ranges[id*3u+2u];
  if(key>=PW_GRID_KEYS || begin>=end || end>b_source[1]) {
    atomicAdd(b_meta[2],1u);atomicOr(b_meta[3],16u);return;
  }
  uint mask=uint(u_hash.y)-1u,start=cellHash(key)&mask,probes=0u,error=4u;
  for(uint i=0u;i<uint(u_hash.z);++i) {
    uint slot=(start+i)&mask,previous;probes=i+1u;
    atomicFetchCompareExchange(b_table[slot*2u],0xffffffffu,key,previous);
    if(previous==0xffffffffu) {
      // Consumers run in a later dispatch, after both key and payload stores.
      b_table[slot*2u+1u]=id;atomicAdd(b_meta[1],1u);error=0u;break;
    }
    if(previous==key) {error=8u;break;}
  }
  atomicAdd(b_metrics[0],probes);atomicMax(b_metrics[1],probes);
  if(error!=0u) {atomicAdd(b_meta[2],1u);atomicOr(b_meta[3],error);}
}
