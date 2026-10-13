$input v_uv, v_color
#include <bgfx_shader.sh>
void main(){
 float rr=dot(v_uv,v_uv);if(rr>1.0)discard;
 gl_FragColor=vec4(v_color.rgb,v_color.a*(1.0-rr));
}
