#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_input,uint,0);
BUFFER_RW(b_output,uint,1);
BUFFER_RW(b_totals,uint,2);
uniform vec4 u_scan;
SHARED uint values[PW_WORKGROUP_SIZE];
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main() {
  uint lane=gl_LocalInvocationID.x;
  uint id=gl_GlobalInvocationID.x;
  uint value=0u;
  if(id<uint(u_scan.x)) value=b_input[id];
  values[lane]=value;
  barrier();
  for(uint stride=1u;stride<PW_WORKGROUP_SIZE;stride*=2u) {
    uint end=(lane+1u)*stride*2u-1u;
    if(end<PW_WORKGROUP_SIZE) values[end]+=values[end-stride];
    barrier();
  }
  if(lane==0u) {b_totals[gl_WorkGroupID.x]=values[PW_WORKGROUP_SIZE-1];values[PW_WORKGROUP_SIZE-1]=0u;}
  barrier();
  for(uint stride=PW_WORKGROUP_SIZE/2u;stride>0u;stride/=2u) {
    uint end=(lane+1u)*stride*2u-1u;
    if(end<PW_WORKGROUP_SIZE) {
      uint previous=values[end-stride];values[end-stride]=values[end];values[end]+=previous;
    }
    barrier();
  }
  if(id<uint(u_scan.x)) b_output[id]=values[lane];
}
