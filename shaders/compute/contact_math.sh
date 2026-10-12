bool validMass(float value) {
  uint bits=floatBitsToUint(value),magnitude=bits&0x7fffffffu;
  return magnitude==0u || ((bits&0x80000000u)==0u && magnitude>=0x00800000u && magnitude<0x7f800000u);
}
float massWeight(float a,float b) {
  // Direct division by FLT_MAX can become a flushed reciprocal on Metal.
  // Normalize exponent/mantissa parts before division; denominator is [1,4).
  uint bitsA=floatBitsToUint(a)&0x7fffffffu,bitsB=floatBitsToUint(b)&0x7fffffffu;
  uint exponentA=bitsA>>23u,exponentB=bitsB>>23u,common=max(exponentA,exponentB);
  float scaledA=uintBitsToFloat((bitsA&0x007fffffu)|0x3f800000u)*exp2(float(int(exponentA)-int(common)));
  float scaledB=0.0;
  if(bitsB!=0u) scaledB=uintBitsToFloat((bitsB&0x007fffffu)|0x3f800000u)*exp2(float(int(exponentB)-int(common)));
  return scaledA/(scaledA+scaledB);
}
vec3 contactNormal(uint query,uint slot,vec3 delta,float squared) {
  if(squared>1e-12) return -delta/sqrt(squared);
  uint axis=(min(query,slot)*1664525u+max(query,slot)*1013904223u)%3u;
  // Explicit components avoid HLSL's dynamic vector-write lowering, which can
  // force unrolling of the surrounding variable-length neighbor loop.
  float sign=query<slot?-1.0:1.0;
  return vec3(axis==0u?sign:0.0,axis==1u?sign:0.0,axis==2u?sign:0.0);
}
bool contactFinite(float value) {return (floatBitsToUint(value)&0x7f800000u)!=0x7f800000u;}
bool contactFinite3(vec3 value) {return contactFinite(value.x) && contactFinite(value.y) && contactFinite(value.z);}
