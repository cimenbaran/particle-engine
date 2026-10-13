#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_result,uvec4,5);
#include "cell_lookup.sh"
#include "spawn_lattice.sh"
#include "sphere_clearance.sh"
uniform vec4 u_spawn_index;
uniform vec4 u_spawn_query;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_spawn_query.x))return;
  if(b_meta[0]!=1u){b_result[id]=uvec4(2u,0u,0u,0u);return;}
  vec3 p;if(!spawnLatticePoint(id,floatBitsToUint(u_spawn_query.y),p)){b_result[id]=uvec4(4u,0u,0u,0u);return;}
  if(p.x< -PW_GRID_EXTENT || p.x>=PW_GRID_EXTENT || p.y<0.0 || p.y>=16.0 || p.z< -PW_GRID_EXTENT || p.z>=PW_GRID_EXTENT){b_result[id]=uvec4(3u,0u,0u,0u);return;}
  ivec3 cell=ivec3(p*8.0)+ivec3(PW_GRID_OFFSET,0,PW_GRID_OFFSET);
  ivec3 lo=max(cell-ivec3(1,1,1),ivec3(0,0,0)),hi=min(cell+ivec3(1,1,1),ivec3(PW_GRID_X-1,PW_GRID_Y-1,PW_GRID_Z-1));
  uvec4 result=uvec4(0u,0u,0u,0u);
  LOOP for(int z=lo.z;z<=hi.z;++z)LOOP for(int y=lo.y;y<=hi.y;++y)LOOP for(int x=lo.x;x<=hi.x;++x) {
    uint key=uint(x+PW_GRID_X*(y+PW_GRID_Y*z)),begin,end,probes;
    uint status=lookupCell(key,uint(u_spawn_index.x),uint(u_spawn_index.y),uint(u_spawn_index.z),begin,end,probes);result.z+=probes;
    if(status==3u){result.x=2u;b_result[id]=result;return;}if(status==0u)continue;
    LOOP for(uint i=begin;i<end;++i) {
      uint slot=b_sorted[i*2u+1u];++result.y;
      if(slot>=uint(u_spawn_index.x) || b_sorted[i*2u]!=key){result.x=2u;b_result[id]=result;return;}
      if(spawnOverlap(b_state[slot*4u].xyz,p))result.x=1u;
    }
  }
  b_result[id]=result;
}
