#include "granular_grid.sh"
#include "cell_hash.sh"
// Requires b_ranges and b_table bindings. Caller must first check readiness.
// Returns 0 absent, 1 found, 3 corrupt payload. Full keys are compared.
uint lookupCell(uint key,uint capacity,uint slots,uint limit,out uint begin,out uint end,out uint probes) {
  begin=0u;end=0u;probes=0u;
  if(key>=PW_GRID_KEYS) return 0u;
  uint mask=slots-1u,start=cellHash(key)&mask;
  LOOP for(uint i=0u;i<limit;++i) {
    uint slot=(start+i)&mask,found=b_table[slot*2u];probes=i+1u;
    if(found==0xffffffffu) return 0u;
    if(found==key) {
      uint range=b_table[slot*2u+1u];
      if(range>=capacity) return 3u;
      uint a=b_ranges[range*3u+1u],b=b_ranges[range*3u+2u];
      if(b_ranges[range*3u]!=key || a>=b || b>capacity) return 3u;
      begin=a;end=b;return 1u;
    }
  }
  // Readiness proves every inserted key fits this same bounded sequence.
  return 0u;
}
