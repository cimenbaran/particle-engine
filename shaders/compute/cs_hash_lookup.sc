#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "cell_hash.sh"
BUFFER_RO(b_ranges,uint,0);
BUFFER_RO(b_table,uint,1);
BUFFER_RO(b_meta,uint,2);
BUFFER_RO(b_keys,uint,3);
BUFFER_RW(b_results,uint,4);
uniform vec4 u_hash;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_hash.w)) return;
  uint begin=0u,end=0u,status=0u,probes=0u,key=b_keys[id];
  if(b_meta[0]!=1u) status=2u;
  else if(key<2097152u) {
    uint mask=uint(u_hash.y)-1u,start=cellHash(key)&mask;
    // A ready table contains every key within this same probe bound.
    // Exhausting it therefore proves absence; an incomplete table never enters.
    for(uint i=0u;i<uint(u_hash.z);++i) {
      uint slot=(start+i)&mask,found=b_table[slot*2u];probes=i+1u;
      if(found==0xffffffffu) break;
      if(found==key) {
        uint range=b_table[slot*2u+1u];status=3u;
        if(range<uint(u_hash.x)) {
          uint a=b_ranges[range*3u+1u],b=b_ranges[range*3u+2u];
          if(b_ranges[range*3u]==key && a<b && b<=uint(u_hash.x)) {begin=a;end=b;status=1u;}
        }
        break;
      }
    }
  }
  b_results[id*4u]=begin;b_results[id*4u+1u]=end;
  b_results[id*4u+2u]=status;b_results[id*4u+3u]=probes;
}
