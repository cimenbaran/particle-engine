const float sphereSkin=0.000002;
// Outward normal and entry time; 2 means no hit in any supported dt.
vec4 sphereImpact(vec3 p,vec3 v,vec4 collider) {
  if(collider.w==0.0)return vec4(0.0,0.0,0.0,2.0);
  float speed=length(v);if(speed==0.0)return vec4(0.0,0.0,0.0,2.0);
  vec3 direction=v/speed,offset=p-collider.xyz;float approach=-dot(offset,direction);
  if(approach<=0.0)return vec4(0.0,0.0,0.0,2.0);
  vec3 perpendicular=offset+direction*approach;float miss=length(perpendicular),radius=collider.w+0.0625;
  if(miss>=radius)return vec4(0.0,0.0,0.0,2.0);
  float chord=sqrt(max(0.0,(radius-miss)*(radius+miss))),distance=length(offset);
  float time=max(0.0,(distance-radius)*(distance+radius)/(speed*(approach+chord)));
  vec3 normal=perpendicular-direction*chord;normal/=length(normal);
  return vec4(normal,min(time,2.0));
}
