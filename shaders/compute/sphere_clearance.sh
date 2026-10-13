#include "sphere_contact.sh"
// Scale before subtraction to preserve signed subnormal particle coordinates.
float clearanceLift(float v) {
  uint bits=floatBitsToUint(v);
  if((bits&0x7f800000u)==0u){float m=float(bits&0x007fffffu)*2.5849394142282115e-26;return (bits&0x80000000u)!=0u?-m:m;}
  return v*18446744073709551616.0;
}
bool spawnOverlap(vec3 p,vec3 q) {
  precise vec3 a=vec3(clearanceLift(p.x),clearanceLift(p.y),clearanceLift(p.z));
  precise vec3 b=vec3(clearanceLift(q.x),clearanceLift(q.y),clearanceLift(q.z));
  precise vec3 d=a-b,bv=a-d,av=d+bv,br=bv-b,ar=a-av,low=ar+br;
  precise vec2 squared=sphereDot(d,d);
  // Diameter squared scaled by 2^128 = 2^122. Neighbor differences <=.25m
  // keep all intermediate products finite; low terms preserve closed tangency.
  precise float gap=(squared.x-5316911983139663491615228241121378304.0)+squared.y+2.0*dot(d,low)+dot(low,low);
  return gap<0.0;
}
