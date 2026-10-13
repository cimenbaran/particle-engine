#include <bgfx_compute.sh>
#include "workgroup.sh"
BUFFER_RO(b_state,vec4,0);
UIMAGE2D_WO(o_result,rgba32ui,1);
uniform vec4 u_blast_export;
NUM_THREADS(PW_WORKGROUP_SIZE,1,1)
void main(){uint slot=gl_GlobalInvocationID.x;if(slot>=uint(u_blast_export.x))return;for(uint lane=0u;lane<4u;++lane){uint index=slot*4u+lane;imageStore(o_result,ivec2(int(index%1024u),int(index/1024u)),floatBitsToUint(b_state[index]));}}
