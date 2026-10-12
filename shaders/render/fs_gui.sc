$input v_uv, v_color
#include <bgfx_shader.sh>
SAMPLER2D(s_texture, 0);
void main() { gl_FragColor = v_color * texture2D(s_texture, v_uv); }
