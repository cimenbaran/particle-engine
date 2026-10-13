const float sphereSkin=0.000002;
// Refine the reciprocal square root before constructing a geometric direction.
// Backend estimates must not leave a non-unit contact normal in friction response.
vec3 sphereUnit(vec3 value) {
  float squared=dot(value,value),inverse=inversesqrt(squared);
  inverse*=1.5-0.5*squared*inverse*inverse;
  return value*inverse;
}
// Compensated products retain the low bits used by near-tangent roots.
// precise is essential: reassociation can erase the residual arithmetic.
vec2 sphereDot(vec3 a,vec3 b) {
  precise vec3 products=a*b;
  precise vec3 errors=vec3(mad(a.x,b.x,-products.x),mad(a.y,b.y,-products.y),mad(a.z,b.z,-products.z));
  precise float sum=products.x+products.y,part=sum-products.x;
  precise float error=(products.x-(sum-part))+(products.y-part);
  precise float total=sum+products.z,tail=total-sum;
  error+=(sum-(total-tail))+(products.z-tail);
  return vec2(total,error+errors.x+errors.y+errors.z);
}
// Outward normal and entry time; 2 means no hit in any supported dt.
vec4 sphereImpact(vec3 p,vec3 v,vec4 collider) {
  if(collider.w==0.0)return vec4(0.0,0.0,0.0,2.0);
  float speed=length(v);if(speed==0.0)return vec4(0.0,0.0,0.0,2.0);
  vec3 direction=sphereUnit(v),offset=p-collider.xyz;float approach=-dot(offset,direction);
  if(approach<=0.0)return vec4(0.0,0.0,0.0,2.0);
  vec3 perpendicular=offset+direction*approach;float miss=length(perpendicular),radius=collider.w+0.0625;
  if(abs(miss-radius)<0.00001) {
    precise vec2 vv=sphereDot(v,v),pv=sphereDot(offset,v),pp=sphereDot(offset,offset);
    precise float rr=radius*radius,c=pp.x-rr;
    precise float cpart=c-pp.x;
    precise float clow=(pp.x-(c-cpart))+(-rr-cpart)+pp.y-mad(radius,radius,-rr);
    precise float bb=pv.x*pv.x,ac=vv.x*c;
    precise float discriminant=(bb-ac)+(mad(pv.x,pv.x,-bb)-mad(vv.x,c,-ac))+2.0*pv.x*pv.y-vv.x*clow-vv.y*c;
    if(discriminant<=0.0)return vec4(0.0,0.0,0.0,2.0);
    precise float root=sqrt(discriminant),time=max(0.0,(c+clow)/(-pv.x-pv.y+root));
    return vec4(sphereUnit(offset+v*time),min(time,2.0));
  }
  if(miss>=radius)return vec4(0.0,0.0,0.0,2.0);
  float chord=sqrt(max(0.0,(radius-miss)*(radius+miss))),distance=length(offset);
  float time=max(0.0,(distance-radius)*(distance+radius)/(speed*(approach+chord)));
  vec3 normal=sphereUnit(perpendicular-direction*chord);
  return vec4(normal,min(time,2.0));
}
