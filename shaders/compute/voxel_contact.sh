#include "sphere_contact.sh"
// Single 0.125m voxel, common particle radius0.0625m. Geometric closed-segment
// query: 0 miss, 1 contact, 2 initial overlap. Validation belongs to the caller.
vec3 voxelSegmentPoint(vec3 start,vec3 end,float time) {
  if(time==0.0)return start;if(time==1.0)return end;
  return mix(start,end,time);
}
float voxelSeparation(vec3 position,vec3 lower) {
  precise vec3 offset=position-clamp(position,lower,lower+vec3_splat(0.125));
  precise vec2 squared=sphereDot(offset,offset);
  precise float separation=(squared.x-0.00390625)+squared.y;return separation;
}
vec3 voxelNormal(vec3 position,vec3 lower) {
  vec3 offset=position-clamp(position,lower,lower+vec3_splat(0.125));
  float squared=dot(offset,offset);
  if(squared>0.0)return offset/sqrt(squared);
  float closest=1000000.0;uint side=0u;
  for(uint candidate=0u;candidate<6u;++candidate) {
    uint axis=candidate/2u;
    float gap=(candidate&1u)!=0u?lower[axis]+0.125-position[axis]:position[axis]-lower[axis];
    if(gap<closest){closest=gap;side=candidate;}
  }
  float sign=(side&1u)!=0u?1.0:-1.0;
  if(side<2u)return vec3(sign,0.0,0.0);
  if(side<4u)return vec3(0.0,sign,0.0);
  return vec3(0.0,0.0,sign);
}
uint voxelSweep(vec3 start,vec3 end,vec3 lower,out vec3 centre,out vec3 normal,out float time) {
  centre=end;normal=vec3_splat(0.0);time=1.0;
  float initial=voxelSeparation(start,lower);
  if(initial<=0.0){time=0.0;centre=start;normal=voxelNormal(start,lower);return initial<0.0?2u:1u;}
  precise vec3 delta=end-start;
  float cuts[8];cuts[0]=0.0;cuts[1]=1.0;uint count=2u;
  for(uint axis=0u;axis<3u;++axis)if(delta[axis]!=0.0) {
    for(uint side=0u;side<2u;++side) {
      precise float plane=lower[axis]+float(side)*0.125;
      precise float cut=(plane-start[axis])/delta[axis];
      if(cut>0.0 && cut<1.0){cuts[count]=cut;++count;}
    }
  }
  LOOP for(uint i=1u;i<count;++i) {
    float value=cuts[i];uint j=i;
    LOOP while(j>0u){if(cuts[j-1u]<=value)break;cuts[j]=cuts[j-1u];--j;}
    cuts[j]=value;
  }
  LOOP for(uint i=1u;i<count;++i) {
    precise float left=cuts[i-1u],duration=cuts[i]-left,middle=left+duration*0.5;
    if(duration<=0.0)continue;
    precise vec3 q=vec3_splat(0.0),v=vec3_splat(0.0);
    for(uint axis=0u;axis<3u;++axis) {
      precise float coordinate=mad(delta[axis],middle,start[axis]);
      if(coordinate<lower[axis] || coordinate>lower[axis]+0.125) {
        precise float plane=coordinate<lower[axis]?lower[axis]:lower[axis]+0.125;
        q[axis]=mad(delta[axis],left,start[axis])-plane;v[axis]=delta[axis];
      }
    }
    precise vec2 qq=sphereDot(q,q),vv=sphereDot(v,v),qv=sphereDot(q,v);
    precise float c=qq.x-0.00390625,part=c-qq.x;
    precise float clow=(qq.x-(c-part))+(-0.00390625-part)+qq.y;
    precise float root=0.0;
    if(c+clow>0.0) {
      if(vv.x==0.0 || qv.x+qv.y>=0.0)continue;
      precise float bb=qv.x*qv.x,ac=vv.x*c;
      precise float discriminant=(bb-ac)+(mad(qv.x,qv.x,-bb)-mad(vv.x,c,-ac))+2.0*qv.x*qv.y-vv.x*clow-vv.y*c;
      if(discriminant<0.0)continue;
      root=(c+clow)/(-qv.x-qv.y+sqrt(discriminant));
    }
    if(root<0.0 || root>duration)continue;
    float candidate=min(1.0,left+root);
    if(candidate==1.0 && voxelSeparation(end,lower)>0.0)continue;
    time=candidate;centre=voxelSegmentPoint(start,end,time);normal=voxelNormal(centre,lower);return 1u;
  }
  return 0u;
}
