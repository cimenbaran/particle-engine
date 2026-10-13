$input a_position
$output v_color
#include <bgfx_shader.sh>
uniform vec4 u_terrain_sphere;
void main() {
  vec3 world=u_terrain_sphere.xyz+a_position*u_terrain_sphere.w;
  gl_Position=mul(u_viewProj,vec4(world,1.0));
  float light=0.25+0.75*max(0.0,dot(a_position,normalize(vec3(-0.4,0.8,0.5))));
  v_color=vec4(vec3(0.32,0.43,0.52)*light,1.0);
}
