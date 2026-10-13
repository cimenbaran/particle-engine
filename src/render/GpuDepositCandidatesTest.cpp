#include "GpuDepositCandidates.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <bit>
#include <cstring>
#include <iostream>
namespace pw {
namespace {template<class T>struct DepositHandle{T value;explicit DepositHandle(T v):value(v){if(!bgfx::isValid(v))throw std::runtime_error("Deposit test allocation failed");}~DepositHandle(){bgfx::destroy(value);}DepositHandle(const DepositHandle&)=delete;};}
void runGpuDepositCandidatesTest(const std::filesystem::path& assets){
 const auto check=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
 const auto rejects=[&](auto f){bool invalid=false;try{f();}catch(const std::invalid_argument&){invalid=true;}check(invalid,"Invalid deposit submit accepted");};
 const std::array materials{terrain::DepositMaterial{2600,true},terrain::DepositMaterial{1600,true},terrain::DepositMaterial{1000,false}};
 terrain::SparseVoxelStore world(terrain::ResidentChunkLayout({-1,0,-1},{2,1,2}),3);
 world.setMaterial(*terrain::addressOf({0,0,0}),0,0);world.setMaterial(*terrain::addressOf({1,0,0}),2,1);
 GpuVoxelWorld occupancy(world,{});
 bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
 DepositHandle exporter(loadCompute(assets,"cs_export"));DepositHandle params(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
 const auto read=[&](bgfx::DynamicVertexBufferHandle input,unsigned count,unsigned laneCount){
  DepositHandle output(bgfx::createTexture2D(std::uint16_t(laneCount),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
  DepositHandle staging(bgfx::createTexture2D(std::uint16_t(laneCount),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
  const float p[]{0,float(count),0,float(laneCount)};bgfx::setUniform(params.value,p);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);if(count)bgfx::dispatch(5,exporter.value,dispatchGroups(count));
  bgfx::TextureRegion a,b;a.handle=output.value;b.handle=staging.value;bgfx::blit(7,b,a);std::vector<std::uint32_t> result(std::max(1u,count)*laneCount*4);const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
 };
 unsigned cases=0;
 for(unsigned capacity:{0u,1u,2u,127u,128u,129u,255u,256u,257u,1024u,4096u})for(unsigned scenario=0;scenario<4;++scenario){
  std::vector<ParticleGpu> particles(std::max(1u,capacity));std::vector<SettlingState> history(std::max(1u,capacity));
  for(unsigned i=0;i<capacity;++i){const unsigned material=(i/12)%3;auto& p=particles[i];p={{-.9375f+float(i%8)*.125f,.0625f,-.9375f+float((i/8)%8)*.125f,.0625f},{0,0,0,terrain::conversionMassProfile(materials[material].density).inverseMass},{293,0,0,0},{material,1,i,1}};history[i]={1,30,0,1};
   switch(i%12){case 0:p.identity[1]=0;break;case 1:history[i].generation=2;break;case 2:break;case 3:history[i].unsupportedTicks=1;break;case 4:p.velocityInvMass[0]=.2f;break;case 5:p.velocityInvMass[3]=0;break;case 6:p.positionRadius[0]=.0625f;p.positionRadius[2]=.0625f;break;case 7:p.positionRadius[0]=.0625f;p.positionRadius[1]=.1875f;p.positionRadius[2]=.0625f;break;case 8:break;case 9:p.positionRadius[0]=.1875f;p.positionRadius[1]=.1875f;p.positionRadius[2]=.0625f;break;case 10:p.positionRadius[1]=1;break;case 11:history[i].quietTicks=29;break;}
  }
  if(capacity && scenario==1){particles[0].identity[0]=0;particles[0].identity[1]=1;particles[0].identity[2]=0;particles[0].identity[3]=0;}
  if(capacity && scenario==2){particles[0].identity[0]=0;particles[0].identity[1]=1;particles[0].identity[2]=0;particles[0].identity[3]=1;particles[0].positionRadius[0]=INFINITY;}
  const double tolerance=scenario==3?.01:.05;
  DepositHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),unsigned(particles.size()*64)),lanes,BGFX_BUFFER_COMPUTE_READ));
  DepositHandle histories(bgfx::createDynamicVertexBuffer(bgfx::copy(history.data(),unsigned(history.size()*16)),lanes,BGFX_BUFFER_COMPUTE_READ));
  GpuDepositCandidates candidates(assets,capacity,materials,tolerance);PoolDiagnostics diagnostic(assets);
  candidates.submit(1,input.value,histories.value,occupancy,world);check(diagnostic.request(6,candidates.status(),0),"Deposit diagnostic request failed");const auto records=read(candidates.output(),capacity,2);
  while(diagnostic.pending()){diagnostic.poll(bgfx::frame());SDL_PumpEvents();}const auto status=diagnostic.latest();
  check(status.active==(capacity && scenario==1?1u:capacity && scenario==2?2u:0u),"Deposit error status wrong");
  if(scenario==0 || scenario==3){unsigned eligible=0,targets=0,solids=0;
   for(unsigned i=0;i<capacity;++i){const auto& p=particles[i];const auto& h=history[i];const bool solid=p.identity[1] && materials[p.identity[0]].solid;
    const bool owner=solid && scenario!=3 && h.generation==p.identity[3] && h.quietTicks==30 && h.unsupportedTicks==0 && p.velocityInvMass[0]==0 && p.velocityInvMass[3]>0;
    terrain::GlobalVoxel cell{};for(unsigned a=0;a<3;++a)cell[a]=std::int64_t(std::floor(double(p.positionRadius[a])*8));
    const auto material=world.materialAt(*terrain::addressOf(cell));bool ready=owner && material && *material==terrain::SparseVoxelStore::empty;
    if(ready && cell[1]>0){auto below=cell;--below[1];const auto support=world.materialAt(*terrain::addressOf(below));ready=support && *support<materials.size() && materials[*support].solid;}
    check(records[i*8+6]==(owner?2u:solid?1u:0u) && records[i*8+7]==unsigned(ready),"Deposit owner/target differs from independent reference");
    if(solid){check(records[i*8+3]==p.identity[0] && records[i*8+4]==i && records[i*8+5]==1,"Deposit identity changed");for(unsigned a=0;a<3;++a)check(std::bit_cast<std::int32_t>(records[i*8+a])==cell[a],"Signed deposit cell wrong");}
    eligible+=owner;targets+=ready;solids+=solid;
   }
   check(status.free==eligible && status.reserved==targets && status.rejected==solids,"Deposit counts wrong");
  }
  const auto after=read(input.value,capacity,4),historyAfter=read(histories.value,capacity,1);
  if(capacity)check(std::memcmp(after.data(),particles.data(),capacity*64)==0 && std::memcmp(historyAfter.data(),history.data(),capacity*16)==0,"Deposit preflight changed source state/history");
  rejects([&]{candidates.submit(UINT16_MAX,input.value,histories.value,occupancy,world);});rejects([&]{candidates.submit(1,input.value,input.value,occupancy,world);});++cases;
 }
 {
  const auto mass=terrain::conversionMassProfile(2600).inverseMass;
  std::array<ParticleGpu,4> input{};std::array<SettlingState,4> h{};
  for(unsigned i=0;i<4;++i){input[i]={{0,.0625f,-.0625f,.0625f},{0,0,0,mass},{293,0,0,0},{0,1,i,1}};h[i]={1,30,0,1};}
  input[0].positionRadius[0]=-std::numeric_limits<float>::denorm_min();input[1].positionRadius[0]=std::numeric_limits<float>::denorm_min();input[2].velocityInvMass[0]=.1f;input[3].velocityInvMass[0]=std::nextafter(.1f,0.f);
  DepositHandle particles(bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),sizeof(input)),lanes,BGFX_BUFFER_COMPUTE_READ));
  DepositHandle histories(bgfx::createDynamicVertexBuffer(bgfx::copy(h.data(),sizeof(h)),lanes,BGFX_BUFFER_COMPUTE_READ));
  GpuDepositCandidates candidates(assets,4,materials);candidates.submit(1,particles.value,histories.value,occupancy,world);const auto records=read(candidates.output(),4,2);
  check(std::bit_cast<std::int32_t>(records[0])==-1 && records[8]==0 && records[6]==2 && records[14]==2,"Subnormal signed cell classification failed");
  check(records[22]==1 && records[30]==1,"Near-threshold speed did not defer conservatively");
  auto other=world.clone();rejects([&]{candidates.submit(1,particles.value,histories.value,occupancy,*other);});rejects([&]{candidates.submit(1,candidates.output(),histories.value,occupancy,world);});
  rejects([&]{GpuDepositCandidates invalid(assets,4097,materials);});rejects([&]{GpuDepositCandidates invalid(assets,4,materials,-.01);});++cases;
 }
 check(world.occupiedVoxelCount()==2 && world.revision()==2,"Deposit preflight changed terrain");
 std::cout<<"GPU deposit candidates passed: "<<cases<<" batches; eligibility, support, signed cells, material mass bounds, malformed-state rejection and immutable source/history\n";
}
}
