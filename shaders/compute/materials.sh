// MaterialGpu v1: mechanical, thermal, linear color, integer identity (4 lanes).
BUFFER_RO(b_materials, vec4, 3);
uint materialColor(uint material) {
  uvec4 c=uvec4(clamp(b_materials[material*4u+2u],0.0,1.0)*255.0+0.5);
  return c.x | (c.y<<8u) | (c.z<<16u) | (c.w<<24u);
}
