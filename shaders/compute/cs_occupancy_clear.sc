#include <bgfx_compute.sh>
BUFFER_RO(b_meta,uint,0);
BUFFER_RO(b_upstream,uint,1);
BUFFER_RW(b_status,uint,2);
uniform vec4 u_occupancy; // capacity, reset only
NUM_THREADS(1,1,1)
void main(){
  b_status[0]=0u;b_status[1]=0u;b_status[2]=0u;b_status[3]=0u;
  if(u_occupancy.y!=0.0)return;
  uint capacity=uint(u_occupancy.x),cells=b_meta[0],records=b_meta[1];
  uint error=b_meta[2]!=0u || cells>records || records>capacity || (cells==0u && records!=0u)?1u:0u;
  if(b_upstream[0]!=0u)error|=2u;
  b_status[0]=error;
  if(error==0u){b_status[1]=cells;b_status[2]=records;}
}
