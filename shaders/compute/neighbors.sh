#include "cell_lookup.sh"
// Lift tiny coordinates before subtraction so Metal's subnormal flushing does
// not erase separations. 2^64 keeps every coordinate in this grid finite.
float liftedCoordinate(float value) {
  uint bits=floatBitsToUint(value);
  if((bits&0x7f800000u)==0u) {
    float magnitude=float(bits&0x007fffffu)*2.5849394142282115e-26; // 2^-85
    return (bits&0x80000000u)!=0u?-magnitude:magnitude;
  }
  return value*18446744073709551616.0; // 2^64
}
bool sameCoordinate(float a,float b) {
  uint x=floatBitsToUint(a),y=floatBitsToUint(b);
  return x==y || ((x|y)&0x7fffffffu)==0u;
}
bool withinSupport(vec3 a,vec3 b,float support,float distanceSquared) {
  uint bits=floatBitsToUint(support)&0x7fffffffu;
  if(bits==0u) return sameCoordinate(a.x,b.x) && sameCoordinate(a.y,b.y) && sameCoordinate(a.z,b.z);
  if(bits<0x2f800000u) { // support < 2^-32; avoid squared-distance underflow
    float scale=liftedCoordinate(support);
    vec3 normalized=vec3(liftedCoordinate(b.x)-liftedCoordinate(a.x),
      liftedCoordinate(b.y)-liftedCoordinate(a.y),liftedCoordinate(b.z)-liftedCoordinate(a.z))/scale;
    return dot(normalized,normalized)<=1.0;
  }
  return distanceSquared<=support*support;
}
// Bind the SAME frozen physical state used to emit/sort the index. All indexed
// materials are solid; liquids, gas, dead slots and the cosmetic pool are absent.
// PW_VISIT_NEIGHBOR(queryIndex,slot,delta,distanceSquared) is the consumer hook.
// Return: accepted count, candidate count (includes self), probes, status.
// Status: 0 success, 1 excluded query, 2 failed index, 3 corrupt/config error.
// Any error invalidates all consumer output, including visits before the error.
uvec4 visitNeighbors(uint queryIndex,vec4 config) {
  uint capacity=uint(config.x),slots=uint(config.y),limit=uint(config.z);
  if(b_meta[0]!=1u) return uvec4(0u,0u,0u,2u);
  uint supportBits=floatBitsToUint(config.w);
  if(((supportBits&0x80000000u)!=0u && (supportBits&0x7fffffffu)!=0u) || (supportBits&0x7f800000u)==0x7f800000u)
    return uvec4(0u,0u,0u,3u);
  uint key=b_sorted[queryIndex*2u],querySlot=b_sorted[queryIndex*2u+1u];
  if(key==0xffffffffu) return uvec4(0u,0u,0u,1u);
  if(key>=2097152u || querySlot>=capacity) return uvec4(0u,0u,0u,3u);
  vec3 position=b_state[querySlot*4u].xyz;
  // Grid diameter is below 32 m. Saturating larger finite supports preserves
  // membership while avoiding squared-distance/float-to-int overflow.
  // Preserve subnormal bits: an arithmetic min may flush them to zero.
  float support=config.w;if(config.w>32.0) support=32.0;
  int radius=int(ceil(min(support*8.0,127.0)));
  if((supportBits&0x7fffffffu)!=0u) radius=max(radius,1);
  ivec3 cell=ivec3(int(key%128u),int((key/128u)%128u),int(key/16384u));
  ivec3 low=max(cell-ivec3(radius,radius,radius),ivec3(0,0,0));
  ivec3 high=min(cell+ivec3(radius,radius,radius),ivec3(127,127,127));
  uint accepted=0u,candidates=0u,probes=0u;
  // Contact callbacks make this body too large for HLSL's automatic unroll.
  // Bounds are runtime grid/support data; retain bounded loops on every axis.
  LOOP for(int z=low.z;z<=high.z;++z)
  LOOP for(int y=low.y;y<=high.y;++y)
  LOOP for(int x=low.x;x<=high.x;++x) {
    uint cellKey=uint(x+128*(y+128*z)),begin,end,cellProbes;
    uint status=lookupCell(cellKey,capacity,slots,limit,begin,end,cellProbes);
    probes+=cellProbes;
    if(status==3u) return uvec4(accepted,candidates,probes,3u);
    if(status==0u) continue;
    LOOP for(uint i=begin;i<end;++i) {
      uint slot=b_sorted[i*2u+1u];++candidates;
      if(slot>=capacity || b_sorted[i*2u]!=cellKey) return uvec4(accepted,candidates,probes,3u);
      if(slot==querySlot) continue;
      vec3 candidatePosition=b_state[slot*4u].xyz;
      vec3 delta=candidatePosition-position;
      float distanceSquared=dot(delta,delta);
      if(withinSupport(position,candidatePosition,support,distanceSquared)) {
        ++accepted;PW_VISIT_NEIGHBOR(queryIndex,slot,delta,distanceSquared);
      }
    }
  }
  return uvec4(accepted,candidates,probes,0u);
}
