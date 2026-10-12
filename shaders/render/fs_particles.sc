$input v_uv, v_color
#include <bgfx_shader.sh>
void main() {
  float rr = dot(v_uv, v_uv);
  if (rr > 1.0) discard;
  gl_FragColor = vec4(v_color.rgb * (0.4 + 0.6 * sqrt(1.0-rr)), 1.0);
}
