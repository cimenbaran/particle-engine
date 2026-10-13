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
  vec3 delta=contact-start;float distance=length(delta),retreat=2e-6;
  LOOP for(uint attempt=0u;attempt<8u;++attempt) {
    fraction=distance>0.0?max(0.0,1.0-retreat/distance):0.0;
    proposed=distance>0.0?start+delta*fraction:start;
    if(voxelMotionClear(proposed))return true;
    retreat*=2.0;
  }
  return false;
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
vec3 voxelContactCone(vec3 velocity,vec3 normals[8],uint count) {
  float squared=dot(velocity,velocity),tolerance=5e-7*max(sqrt(squared),.001);
  if(voxelConeFeasible(velocity,normals,count,tolerance))return velocity;
  vec3 best=vec3_splat(0.0);float error=squared;
  LOOP for(uint i=0u;i<count;++i) {
    vec3 candidate=velocity-normals[i]*dot(velocity,normals[i]);
    float difference=dot(candidate-velocity,candidate-velocity);
    if(difference<error && voxelConeFeasible(candidate,normals,count,tolerance)){best=candidate;error=difference;}
    LOOP for(uint j=0u;j<i;++j) {
      vec3 axis=cross(normals[i],normals[j]);float lengthSquared=dot(axis,axis);
      if(lengthSquared<=1e-12)continue;
      candidate=axis*(dot(velocity,axis)/lengthSquared);difference=dot(candidate-velocity,candidate-velocity);
      if(difference<error && voxelConeFeasible(candidate,normals,count,tolerance)){best=candidate;error=difference;}
    }
  }
  return best;
}
