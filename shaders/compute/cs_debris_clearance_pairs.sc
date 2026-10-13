#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_queries,vec4,5);
#include "cell_lookup.sh"
#include "sphere_clearance.sh"
uniform vec4 u_spawn_index;
uniform vec4 u_debris_clearance;
int queryCell(float v){uint bits=floatBitsToUint(v);if((bits&0x7f800000u)==0u)return (bits&0x80000000u)!=0u && (bits&0x007fffffu)!=0u?-1:0;return int(floor(v*8.0));}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_debris_clearance.y))return;
  uvec4 result=floatBitsToUint(b_queries[id*3u+2u]);
  if(b_meta[0]!=1u)result.x|=4u;
  if(result.x!=0u){b_queries[id*3u+2u]=uintBitsToFloat(result);return;}
  vec3 p=b_queries[id*3u].xyz;uint self=floatBitsToUint(b_queries[id*3u+1u]).x;
  ivec3 cell=ivec3(queryCell(p.x)+PW_GRID_OFFSET,queryCell(p.y),queryCell(p.z)+PW_GRID_OFFSET);
  ivec3 lo=max(cell-ivec3(1,1,1),ivec3(0,0,0)),hi=min(cell+ivec3(1,1,1),ivec3(PW_GRID_X-1,PW_GRID_Y-1,PW_GRID_Z-1));
  LOOP for(int z=lo.z;z<=hi.z;++z)LOOP for(int y=lo.y;y<=hi.y;++y)LOOP for(int x=lo.x;x<=hi.x;++x) {
    uint key=uint(x+PW_GRID_X*(y+PW_GRID_Y*z)),begin,end,probes;
    uint status=lookupCell(key,uint(u_spawn_index.x),uint(u_spawn_index.y),uint(u_spawn_index.z),begin,end,probes);result.z+=probes;
    if(status==3u){result.x|=4u;b_queries[id*3u+2u]=uintBitsToFloat(result);return;}if(status==0u)continue;
    LOOP for(uint i=begin;i<end;++i) {
      uint slot=b_sorted[i*2u+1u];
      if(slot>=uint(u_spawn_index.x) || b_sorted[i*2u]!=key){result.x|=4u;b_queries[id*3u+2u]=uintBitsToFloat(result);return;}
      if(slot==self)continue;++result.y;
      if(spawnOverlap(b_state[slot*4u].xyz,p))result.x|=8u;
    }
  }
  b_queries[id*3u+2u]=uintBitsToFloat(result);
}
