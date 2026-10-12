$input a_position, i_data0, i_data1
$output v_uv, v_color
#include <bgfx_shader.sh>
void main() {
  vec4 center = mul(u_view, vec4(i_data0.xyz, 1.0));
  center.xy += a_position.xy * i_data0.w;
  gl_Position = mul(u_proj, center);
  v_uv = a_position.xy;
  uint packed = floatBitsToUint(i_data1.x);
  v_color = vec4(float(packed & 255u), float((packed >> 8u) & 255u),
                 float((packed >> 16u) & 255u), float((packed >> 24u) & 255u)) / 255.0;
}
