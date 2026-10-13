#include "voxel_contact.sh"
// Caller provides read-only uvec4 b_world with the immutable resident header.
// enteringOnly excludes non-entering tangency/contact for motion response.
uint voxelWorldSweep(vec3 start,vec3 end,bool enteringOnly,uint excluded,out vec3 centre,out vec3 normal,
                     out float time,out uint material,out uint index) {
  uint kind=0u;material=0xffffffffu;index=0xffffffffu;
  centre=end;normal=vec3_splat(0.0);time=1.0;
  if(!contactFinite3(start) || !contactFinite3(end) ||
     any(greaterThan(abs(start),vec3_splat(64.0))) || any(greaterThan(abs(end),vec3_splat(64.0))) ||
     any(greaterThan(abs(end-start),vec3_splat(0.125)))) {
    kind=3u;centre=vec3_splat(0.0);
  } else {
    ivec3 minimum=ivec3(b_world[0].xyz),dimensions=ivec3(b_world[1].xyz);
    vec3 lower=vec3(minimum)*0.125,upper=lower+vec3(dimensions)*4.0;
    vec3 low=min(start,end)-vec3_splat(0.0625),high=max(start,end)+vec3_splat(0.0625);
    // Compare centres to inset bounds: adding the radius to a centre just
    // outside can round back onto the boundary and conceal an exit.
    if(any(lessThan(min(start,end),lower+vec3_splat(0.0625))) ||
       any(greaterThan(max(start,end),upper-vec3_splat(0.0625))))kind=4u;
    else {
      // Include both sides of exact grid planes for closed tangency. At most
      // four candidates per axis for the bounded segment, with no truncation.
      ivec3 first=max(ivec3(ceil(low*8.0))-ivec3(1,1,1)-minimum,ivec3(0,0,0));
      ivec3 last=min(ivec3(floor(high*8.0))-minimum,dimensions*32-ivec3(1,1,1));
      LOOP for(int z=first.z;z<=last.z;++z) {
        LOOP for(int y=first.y;y<=last.y;++y) {
          LOOP for(int x=first.x;x<=last.x;++x) {
            ivec3 voxel=ivec3(x,y,z),chunk=voxel/32,local=voxel-chunk*32;
            uint slot=uint(chunk.x+dimensions.x*(chunk.y+dimensions.y*chunk.z));
            uint address=slot*32768u+uint(local.x+32*(local.y+32*local.z));
            uvec4 packed=b_world[4u+address/4u];uint found=packed[address%4u];
            if(found==0xffffffffu)continue;
            vec3 candidateCentre,candidateNormal;float candidateTime;
            uint candidate=voxelSweep(start,end,vec3(minimum+voxel)*0.125,candidateCentre,candidateNormal,candidateTime);
            if(candidate==1u && address==excluded)continue;
            if(enteringOnly && candidate==1u && dot(end-start,candidateNormal)>=0.0)continue;
            // Overlap takes priority at t0; remaining exact ties use lowest
            // chunk-major voxel address, independent of traversal order.
            bool earlier=candidateTime<time || (candidateTime==time &&
              ((candidate==2u && kind!=2u) || (candidate==kind && address<index)));
            if(candidate!=0u && (kind==0u || earlier)) {
              kind=candidate;time=candidateTime;normal=candidateNormal;centre=candidateCentre;index=address;material=found;
            }
          }
        }
      }
      if(!contactFinite3(centre) || !contactFinite3(normal) || !contactFinite(time) || time<0.0 || time>1.0) {
        kind=3u;centre=vec3_splat(0.0);normal=vec3_splat(0.0);time=1.0;index=0xffffffffu;material=0xffffffffu;
      }
    }
  }
  return kind;
}
