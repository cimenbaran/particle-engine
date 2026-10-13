// Shared command layout. Site count is bit-preserving; coordinates are exact eighths.
uniform vec4 u_spawn_lattice_min;
uniform vec4 u_spawn_lattice_size;
uint spawnLatticeHash(uint x){x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
bool spawnLatticePoint(uint request,uint seed,out vec3 position) {
  uint sites=floatBitsToUint(u_spawn_lattice_size.w);position=vec3_splat(0.0);
  if(request>=sites)return false;
  uvec3 dimensions=uvec3(u_spawn_lattice_size.xyz);
  uint index=(spawnLatticeHash(seed)%sites+request)%sites;
  uvec3 cell=uvec3(index%dimensions.x,(index/dimensions.x)%dimensions.y,index/(dimensions.x*dimensions.y));
  position=(u_spawn_lattice_min.xyz+vec3(cell))*.125;return true;
}
