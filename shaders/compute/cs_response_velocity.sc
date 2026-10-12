#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
BUFFER_RO(b_sorted,uint,1);
BUFFER_RO(b_ranges,uint,2);
BUFFER_RO(b_table,uint,3);
BUFFER_RO(b_meta,uint,4);
BUFFER_RW(b_result,vec4,5);
uniform vec4 u_neighbors;
uniform vec4 u_response;
#include "response_pair.sh"
void responseError(uint query,uint error) {
  uvec4 info=floatBitsToUint(b_result[query*2u+1u]);info.x|=error;b_result[query*2u+1u]=uintBitsToFloat(info);
}
void respond(uint queryIndex,uint slot,vec3 delta,float squared) {
  uint query=b_sorted[queryIndex*2u+1u];
  uint error=floatBitsToUint(b_state[slot*8u+7u].y);
  if(error!=0u) {responseError(query,error);return;}
  if(!responsePair(query,slot,squared)) return;
  float a=b_state[query*8u+1u].w,b=b_state[slot*8u+1u].w;if(a==0.0) return;
  if(squared<=1e-12) {delta=b_state[slot*8u+4u].xyz-b_state[query*8u+4u].xyz;squared=dot(delta,delta);}
  vec3 normal=contactNormal(query,slot,delta,squared);
  vec3 relative=b_state[query*8u+1u].xyz-b_state[slot*8u+1u].xyz;
  vec3 incoming=b_state[query*8u+5u].xyz-b_state[slot*8u+5u].xyz;
  float cn=dot(relative,normal),vn=dot(incoming,normal),normalChange=0.0,support=0.0;
  vec4 materialA=b_state[query*8u+6u],materialB=b_state[slot*8u+6u];
  if(!contactFinite3(relative) || !contactFinite3(incoming) || !contactFinite(cn) || !contactFinite(vn)) {responseError(query,8u);return;}
  if(vn<=0.0) {
    float target=vn< -u_response.w?-min(materialA.z,materialB.z)*vn:0.0;
    normalChange=target-cn;support=max(abs(cn-vn),target-vn);
  } else if(cn<0.0) {normalChange=-cn;support=normalChange;}
  vec3 tangent=relative-cn*normal;float speed=sqrt(dot(tangent,tangent));
  if(!contactFinite(normalChange) || !contactFinite(support) || !contactFinite(speed) || !contactFinite3(tangent)) {responseError(query,8u);return;}
  float friction=min(speed,sqrt(materialA.w*materialB.w)*support);
  vec3 change=normal*normalChange;if(speed>0.0) change-=tangent*(friction/speed);
  uint degree=max(floatBitsToUint(b_state[query*8u+7u].x),b>0.0?floatBitsToUint(b_state[slot*8u+7u].x):0u);
  if(degree==0u) {responseError(query,2u);return;}
  vec4 accumulated=b_result[query*2u];accumulated.xyz+=change*(massWeight(a,b)/float(degree));
  b_result[query*2u]=accumulated;
}
#define PW_STATE_STRIDE 8u
#define PW_VISIT_NEIGHBOR(_query,_slot,_delta,_distance) respond(_query,_slot,_delta,_distance)
#include "neighbors.sh"
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint id=gl_GlobalInvocationID.x;if(id>=uint(u_response.x)) return;
  if(b_meta[0]!=1u) {
    b_result[id*2u]=vec4(0.0,0.0,0.0,0.0);b_result[id*2u+1u]=uintBitsToFloat(uvec4(2u,0u,0u,0u));return;
  }
  uint slot=b_sorted[id*2u+1u];uvec4 degree=floatBitsToUint(b_state[slot*8u+7u]);
  b_result[slot*2u]=vec4(0.0,0.0,0.0,0.0);b_result[slot*2u+1u]=uintBitsToFloat(uvec4(degree.y,degree.x,degree.z,degree.w));
  if(degree.y!=0u) return;
  uvec4 result=visitNeighbors(id,u_neighbors);
  if(result.w>1u) responseError(slot,2u);
  if(result.w==0u) {
    vec3 velocity=b_state[slot*8u+1u].xyz+b_result[slot*2u].xyz;
    if(!contactFinite3(velocity)) responseError(slot,8u);
    b_result[slot*2u]=vec4(velocity,0.0);
  }
}
