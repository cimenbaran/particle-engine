#include <bgfx_compute.sh>
BUFFER_RW(b_state, vec4, 0);
BUFFER_RO(b_seeds, vec4, 1);
BUFFER_RW(b_counts, uint, 2);
uniform vec4 u_cosmetic;
NUM_THREADS(1,1,1)
void main(){
 uint cursorDust=0u,cursorSpark=uint(u_cosmetic.y);
 for(uint seed=0u;seed<uint(u_cosmetic.w);++seed){
  uint kind=floatBitsToUint(b_seeds[seed*4u+3u]).x;
  uint end=kind==0u?uint(u_cosmetic.y):uint(u_cosmetic.x);
  uint slot=kind==0u?cursorDust:cursorSpark;
  for(;slot<end;++slot){uvec4 old=floatBitsToUint(b_state[slot*4u+3u]);if(old.y==0u && old.w!=0xffffffffu)break;}
  if(kind==0u)cursorDust=slot;else cursorSpark=slot;
  if(slot==end){b_counts[2u+kind]=min(b_counts[2u+kind],0xfffffffeu)+1u;continue;}
  uvec4 old=floatBitsToUint(b_state[slot*4u+3u]);
  for(uint lane=0u;lane<3u;++lane)b_state[slot*4u+lane]=b_seeds[seed*4u+lane];
  b_state[slot*4u+3u]=uintBitsToFloat(uvec4(kind,1u,slot,old.w+1u));
  if(kind==0u)cursorDust=slot+1u;else cursorSpark=slot+1u;
 }
}
