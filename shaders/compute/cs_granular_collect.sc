#include <bgfx_compute.sh>
BUFFER_RO(b_source,uint,0);
BUFFER_RW(b_step,uint,1);
BUFFER_RW(b_neighbors,uint,2);
uniform vec4 u_collect; // phase bit, metric mode (1 boundary, 2 contacts)
NUM_THREADS(1,1,1)
void main() {
  if(b_source[0]!=0u) b_step[0]|=uint(u_collect.x);
  if(u_collect.y==1.0) {b_step[1]+=b_source[1];b_step[3]+=b_source[2];}
  if(u_collect.y==2.0) {
    b_step[2]=max(b_step[2],b_source[1]);
    if(b_source[0]!=0u)b_neighbors[0]|=uint(u_collect.x);
    b_neighbors[1]=max(b_neighbors[1],b_source[2]);
    b_neighbors[2]=max(b_neighbors[2],b_source[3]);
    b_neighbors[3]+=1u;
  }
}
