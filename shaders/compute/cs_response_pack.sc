#include <bgfx_compute.sh>
#include "workgroup.sh"
#include "contact_math.sh"
BUFFER_RO(b_predicted,vec4,0);
BUFFER_RO(b_current,vec4,1);
BUFFER_RO(b_materials,vec4,2);
BUFFER_RO(b_upstream,uint,3);
BUFFER_RW(b_packed,vec4,4);
BUFFER_RW(b_status,uint,5);
uniform vec4 u_response; // capacity, padded displacement bound, slop, bounce threshold
uniform vec4 u_response_materials; // material count
bool validCoefficient(float value) {
  uint bits=floatBitsToUint(value),magnitude=bits&0x7fffffffu;
  return magnitude==0u || ((bits&0x80000000u)==0u && magnitude<=0x3f800000u);
}
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_response.x)) return;
  uint base=id*4u,outBase=id*8u;
  for(uint i=0u;i<4u;++i) b_packed[outBase+i]=b_current[base+i];
  b_packed[outBase+4u]=b_predicted[base];b_packed[outBase+5u]=b_predicted[base+1u];
  b_packed[outBase+7u]=vec4(0.0,0.0,0.0,0.0);
  uint error=b_upstream[0]==0u?0u:1u,eligible=0u;vec2 coefficients=vec2(0.0,0.0);
  uvec4 identity=floatBitsToUint(b_current[base+3u]);
  if(error==0u && (identity.y&1u)!=0u) {
    if(identity.x>=uint(u_response_materials.x)) error=4u;
    else {
      uint phase=floatBitsToUint(b_materials[identity.x*4u+3u]).y;
      if(phase>2u) error=4u;
      else if(phase==0u) {
        eligible=1u;uvec4 oldIdentity=floatBitsToUint(b_predicted[base+3u]);
        vec3 displacement=b_current[base].xyz-b_predicted[base].xyz;
        vec4 mechanical=b_materials[identity.x*4u];coefficients=mechanical.zy;
        if((identity.x!=oldIdentity.x || identity.y!=oldIdentity.y || identity.z!=oldIdentity.z || identity.w!=oldIdentity.w) || identity.z!=id || identity.w==0u ||
           b_current[base].w!=0.0625 || b_predicted[base].w!=0.0625 ||
           floatBitsToUint(b_current[base+1u].w)!=floatBitsToUint(b_predicted[base+1u].w) || !validMass(b_current[base+1u].w) ||
           !contactFinite3(b_current[base].xyz) || !contactFinite3(b_predicted[base].xyz) ||
           !contactFinite3(b_current[base+1u].xyz) || !contactFinite3(b_predicted[base+1u].xyz) ||
           !contactFinite(dot(displacement,displacement)) || dot(displacement,displacement)>u_response.y*u_response.y ||
           !validCoefficient(coefficients.x) || !validCoefficient(coefficients.y)) error=4u;
      }
    }
  }
  b_packed[outBase+6u]=vec4(uintBitsToFloat(error),uintBitsToFloat(eligible),coefficients.x,coefficients.y);
  if(error!=0u) atomicOr(b_status[0],error);
}
