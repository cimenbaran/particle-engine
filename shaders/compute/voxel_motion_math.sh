// Shared checked separation and contact-cone geometry. Requires b_world.
bool voxelMotionClear(vec3 position) {
  vec3 centre,normal;float time;uint material,index;
  return voxelWorldSweep(position,position,false,0xffffffffu,centre,normal,time,material,index)<=1u;
}
// A normal skin can cross a neighboring surface in a narrow gap. Prefer that
// skin when valid; otherwise retreat along the already swept, contained path.
// The original segment start is a valid fallback even for a t=0 impact.
bool voxelMotionSeparate(vec3 start,vec3 contact,vec3 normal,out vec3 proposed,out float fraction) {
  fraction=1.0;proposed=contact+normal*2e-6;
  if(voxelMotionClear(proposed))return true;
  // Exact contact can be clear even when a skin crosses the opposite side of
  // a narrow gap. Keep this representable surface point before retreating.
  proposed=contact;if(voxelMotionClear(proposed))return true;
  vec3 delta=contact-start;float distance=length(delta),retreat=2e-6;
  LOOP for(uint attempt=0u;attempt<8u;++attempt) {
    fraction=distance>0.0?max(0.0,1.0-retreat/distance):0.0;
    proposed=distance>0.0?start+delta*fraction:start;
    if(voxelMotionClear(proposed))return true;
    retreat*=2.0;
  }
  proposed=start;fraction=0.0;return voxelMotionClear(start);
}
vec3 voxelMotionLower(uint index) {
  uint chunk=index/32768u,local=index%32768u;uvec3 dims=b_world[1].xyz;
  ivec3 cell=ivec3(int(chunk%dims.x)*32+int(local%32u),int((chunk/dims.x)%dims.y)*32+int((local/32u)%32u),int(chunk/(dims.x*dims.y))*32+int(local/1024u));
  return vec3(ivec3(b_world[0].xyz)+cell)*.125;
}
bool voxelConeFeasible(vec3 velocity,vec3 normals[8],uint count,float tolerance) {
  LOOP for(uint i=0u;i<count;++i)if(dot(velocity,normals[i])< -tolerance)return false;
  return true;
}
// Euclidean projection onto the contact cone. In 3D an optimum is the input,
// a single-plane projection, a two-plane intersection, or zero. Enumerating
// these avoids slowly converging alternating projections in narrow wedges.
float voxelContactSpeed(vec3 v) {
  float largest=max(abs(v.x),max(abs(v.y),abs(v.z)));
  return largest>0.0?largest*length(v/largest):0.0;
}
vec3 voxelContactCone(vec3 velocity,vec3 normals[8],uint count) {
  float speed=voxelContactSpeed(velocity);
  if(voxelConeFeasible(velocity,normals,count,5e-7*speed))return velocity;
  vec3 best=vec3_splat(0.0);float retained=0.0;
  LOOP for(uint i=0u;i<count;++i) {
    vec3 candidate=cross(normals[i],cross(velocity,normals[i]))/dot(normals[i],normals[i]);
    float candidateSpeed=voxelContactSpeed(candidate);
    if(candidateSpeed>retained && voxelConeFeasible(candidate,normals,count,5e-7*candidateSpeed)){best=candidate;retained=candidateSpeed;}
    LOOP for(uint j=0u;j<i;++j) {
      vec3 axis=cross(normals[i],normals[j]);float lengthSquared=dot(axis,axis);
      if(lengthSquared<=1e-12)continue;
      candidate=axis*(dot(velocity,axis)/lengthSquared);candidateSpeed=voxelContactSpeed(candidate);
      if(candidateSpeed>retained && voxelConeFeasible(candidate,normals,count,5e-7*candidateSpeed)){best=candidate;retained=candidateSpeed;}
    }
  }
  return best;
}
