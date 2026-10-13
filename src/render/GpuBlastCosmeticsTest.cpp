#include "GpuBlastCosmetics.hpp"
#include "Shaders.hpp"
#include "PoolDiagnostics.hpp"
#include "VoxelProbes.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <bit>
#include <cstring>
#include <iostream>
namespace pw {
namespace {template<class T>struct CosmeticHandle{T value;explicit CosmeticHandle(T v):value(v){if(!bgfx::isValid(v))throw std::runtime_error("Cosmetic test allocation failed");}~CosmeticHandle(){bgfx::destroy(value);}CosmeticHandle(const CosmeticHandle&)=delete;};}
void runGpuBlastCosmeticsTest(const std::filesystem::path& assets){
 const auto check=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
 const auto rejects=[&](auto f){bool failed=false;try{f();}catch(const std::invalid_argument&){failed=true;}check(failed,"Invalid cosmetic request accepted");};
 CosmeticHandle exporter(loadCompute(assets,"cs_export"));CosmeticHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
 const auto read=[&](bgfx::DynamicVertexBufferHandle state,unsigned count,unsigned lanes=4){
  CosmeticHandle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
  CosmeticHandle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
  const float p[]{0,float(count),0,float(lanes)};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,exporter.value,dispatchGroups(count));
  bgfx::TextureRegion a,b;a.handle=image.value;b.handle=staging.value;bgfx::blit(7,b,a);std::vector<ParticleGpu> result(count);const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
 };
 PoolDiagnostics diagnostics(assets);
 const auto counts=[&](GpuBlastCosmetics& effects){check(diagnostics.request(6,effects.counts(),0),"Cosmetic count request failed");while(diagnostics.pending()){diagnostics.poll(bgfx::frame());SDL_PumpEvents();}return diagnostics.latest();};
 auto materials=loadMaterials(assets/"materials/default.json");VoxelProbes physical(assets,materials,true,true);auto original=read(physical.state(),physical.capacity());
 // Give physical storage live nonzero bytes to make isolation checks meaningful.
 original[0]={{0,3,0,.0625f},{1,2,3,1},{293,0,0,0},{0,1,0,1}};bgfx::update(physical.state(),0,bgfx::copy(original.data(),unsigned(original.size()*64)));original=read(physical.state(),physical.capacity());
 BlastCosmeticEvent event{{-8,4,-4},.5f,42};unsigned cases=0;
 for(auto capacities:{std::array<unsigned,2>{0,1},{1,0},{1,1},{127,129},{128,256},{512,256},{2048,2048}}){
  GpuBlastCosmetics effects(assets,capacities[0],capacities[1]);const auto cap=effects.capacity();
  const auto empty=counts(effects);check(empty.active==0 && empty.free==0 && empty.reserved==0 && empty.rejected==0,"Cosmetic reset counts wrong");
  const auto seeds=planBlastCosmetics(event,{128,64});effects.emit(1,event,{128,64});auto snapshot=counts(effects);auto state=read(effects.state(),cap);auto instances=read(effects.instances(),cap,2);
  const unsigned dust=std::min(128u,capacities[0]),spark=std::min(64u,capacities[1]);
  check(snapshot.active==dust && snapshot.free==spark && snapshot.reserved==128-dust && snapshot.rejected==64-spark,"Cosmetic independent capacity accounting failed");
  for(unsigned i=0;i<cap;++i){const bool isSpark=i>=capacities[0];const unsigned rank=isSpark?i-capacities[0]:i;const bool live=rank<(isSpark?spark:dust);const auto& p=state[i];
   check(bool(p.identity[1])==live,"Cosmetic slot membership wrong");if(!live)continue;const auto& seed=seeds[(isSpark?128:0)+rank];
   check(p.identity[0]==unsigned(isSpark) && p.identity[2]==i && p.identity[3]==1 && p.velocityInvMass[3]==0 && p.thermalState[3]==0 && p.thermalState[2]==seed.lifetime,"Cosmetic identity/lifetime/mass wrong");
   for(unsigned a=0;a<3;++a)check(p.positionRadius[a]==seed.position[a] && p.velocityInvMass[a]==seed.velocity[a],"Cosmetic seed upload changed");
   check(p.positionRadius[3]==seed.radius && std::memcmp(p.positionRadius,instances[i].positionRadius,16)==0 && std::bit_cast<unsigned>(instances[i].velocityInvMass[3])==1,"Cosmetic instance publication failed");
  }
  effects.step(1);auto moved=read(effects.state(),cap);instances=read(effects.instances(),cap,2);
  for(unsigned i=0;i<cap;++i)if(state[i].identity[1]){const bool sparkKind=state[i].identity[0]==1;const double dt=1./60;
   for(unsigned a=0;a<3;++a){const double v=(state[i].velocityInvMass[a]+(a==1?(sparkKind?-9.81:.4)*dt:0))*(1-(sparkKind?.3:1.8)*dt);const double p=state[i].positionRadius[a]+v*dt;check(std::abs(moved[i].velocityInvMass[a]-v)<2e-6 && std::abs(moved[i].positionRadius[a]-p)<2e-6,"Cosmetic trajectory differs from CPU oracle");}
   const unsigned alpha=std::bit_cast<unsigned>(instances[i].velocityInvMass[0])>>24;check(alpha<(sparkKind?255u:120u),"Cosmetic alpha did not fade");
  }
  for(unsigned i=0;i<121;++i)effects.step(1);snapshot=counts(effects);state=read(effects.state(),cap);
  check(snapshot.active==0 && snapshot.free==0 && snapshot.reserved==128-dust && snapshot.rejected==64-spark,"Cosmetic lifetime or retained drop counts failed");
  effects.emit(1,event,{128,64});state=read(effects.state(),cap);for(const auto& p:state)if(p.identity[1])check(p.identity[3]==2,"Cosmetic slot generation did not advance");
  // Fill both partitions, then prove a full-pool burst leaves every live record unchanged.
  for(unsigned i=0;i<32;++i)effects.emit(1,event,{128,64});const auto full=read(effects.state(),cap);effects.emit(1,event,{128,64});const auto unchanged=read(effects.state(),cap);
  check(std::memcmp(full.data(),unchanged.data(),cap*64)==0,"Overflow changed existing cosmetic records");
  effects.reset(1);snapshot=counts(effects);state=read(effects.state(),cap);check(snapshot.active==0 && snapshot.free==0 && snapshot.reserved==0 && snapshot.rejected==0,"Cosmetic reset retained counts");
  for(const auto& p:state){const ParticleGpu zero{};check(std::memcmp(&p,&zero,64)==0,"Cosmetic reset retained slot state");}++cases;
 }
 {
  GpuBlastCosmetics effects(assets,1,1);read(effects.state(),2);std::array<ParticleGpu,2> retired{};retired[0].identity[3]=UINT32_MAX;bgfx::update(effects.state(),0,bgfx::copy(retired.data(),sizeof(retired)));
  effects.emit(1,event,{1,1});auto result=counts(effects);check(result.active==0 && result.free==1 && result.reserved==1 && result.rejected==0,"Exhausted cosmetic generation wrapped or stole other family capacity");
  rejects([&]{effects.step(1,-1);});rejects([&]{effects.step(1,INFINITY);});rejects([&]{effects.step(UINT16_MAX);});rejects([&]{effects.emit(1,event,{129,0});});rejects([&]{GpuBlastCosmetics invalid(assets,0,0);});rejects([&]{GpuBlastCosmetics invalid(assets,2049,0);});++cases;
 }
 {
  GpuBlastCosmetics effects(assets,1,0);read(effects.state(),1);
  CosmeticHandle target(bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_RT|BGFX_TEXTURE_BLIT_DST));
  CosmeticHandle framebuffer(bgfx::createFrameBuffer(1,&target.value,false));
  CosmeticHandle staging(bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
  const float identity[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  const auto capture=[&](float age){
   const ParticleGpu particle{{0,0,.5f,.5f},{0,0,0,0},{0,0,2,age},{0,1,0,1}};
   bgfx::update(effects.state(),0,bgfx::copy(&particle,64));effects.step(1,0);
   bgfx::setViewFrameBuffer(10,framebuffer.value);bgfx::setViewRect(10,0,0,64,64);bgfx::setViewTransform(10,identity,identity);bgfx::setViewClear(10,BGFX_CLEAR_COLOR,0x000000ff);bgfx::touch(10);effects.render(10);
   bgfx::TextureRegion a,b;a.handle=target.value;b.handle=staging.value;bgfx::blit(11,b,a);std::vector<std::uint8_t> pixels(64*64*4);const auto ready=bgfx::read(b,pixels.data());while(bgfx::frame()<ready)SDL_PumpEvents();
   std::uint64_t sum=0;for(unsigned i=0;i<pixels.size();i+=4)sum+=pixels[i]+pixels[i+1]+pixels[i+2];return sum;
  };
  const auto initial=capture(0),half=capture(1),expired=capture(2);
  check(initial>1000 && half>initial*.4 && half<initial*.6 && expired==0,"Cosmetic offscreen rendering/fade/expiry failed");
  bgfx::setViewFrameBuffer(10,BGFX_INVALID_HANDLE);bgfx::setViewClear(10,BGFX_CLEAR_NONE);++cases;
 }
 const auto physicalAfter=read(physical.state(),physical.capacity());check(std::memcmp(original.data(),physicalAfter.data(),original.size()*64)==0,"Cosmetic GPU work changed physical storage");
 std::cout<<"GPU blast cosmetics passed: "<<cases<<" scenarios; independent capacities, exact uploads, CPU trajectories, fade, expiry, overflow, reuse, reset and physical byte isolation\n";
}
}
