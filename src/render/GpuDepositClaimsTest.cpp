#include "GpuDepositClaims.hpp"
#include "Shaders.hpp"
#include "PoolDiagnostics.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <bit>
#include <set>
#include <random>
#include <iostream>
namespace pw {
namespace {template<class T>struct ClaimHandle{T value;explicit ClaimHandle(T v):value(v){if(!bgfx::isValid(v))throw std::runtime_error("Deposit claim test allocation failed");}~ClaimHandle(){bgfx::destroy(value);}ClaimHandle(const ClaimHandle&)=delete;};}
void runGpuDepositClaimsTest(const std::filesystem::path& assets){
 const auto check=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
 const auto rejects=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}check(bad,"Invalid deposit claim submit accepted");};
 const std::array materials{terrain::DepositMaterial{2600,true},terrain::DepositMaterial{1600,true}};
 terrain::SparseVoxelStore world(terrain::ResidentChunkLayout({-4,0,-4},{8,1,8}),2);GpuVoxelWorld occupancy(world,{});
 bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
 ClaimHandle exporter(loadCompute(assets,"cs_export"));ClaimHandle params(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
 ClaimHandle scanExport(loadCompute(assets,"cs_scan_export"));ClaimHandle scanParams(bgfx::createUniform("u_scan",bgfx::UniformType::Vec4));
 const auto read=[&](bgfx::DynamicVertexBufferHandle input,unsigned count,unsigned laneCount){
  ClaimHandle image(bgfx::createTexture2D(std::uint16_t(laneCount),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
  ClaimHandle staging(bgfx::createTexture2D(std::uint16_t(laneCount),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
  const float p[]{0,float(count),0,float(laneCount)};bgfx::setUniform(params.value,p);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);if(count)bgfx::dispatch(5,exporter.value,dispatchGroups(count));
  bgfx::TextureRegion a,b;a.handle=image.value;b.handle=staging.value;bgfx::blit(7,b,a);std::vector<std::uint32_t> result(std::max(1u,count)*laneCount*4);const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
 };
 const auto readOwners=[&](GpuDepositClaims& claims,unsigned count){const auto height=std::uint16_t(std::max(1u,(count+255)/256));
  ClaimHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));ClaimHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
  const float p[]{float(count),0,0,0};bgfx::setUniform(scanParams.value,p);bgfx::setBuffer(0,claims.owners(),bgfx::Access::Read);bgfx::setBuffer(1,claims.status(),bgfx::Access::Read);bgfx::setBuffer(2,claims.owners(),bgfx::Access::Read);bgfx::setImage(3,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,scanExport.value,dispatchGroups(std::max(1u,count)));
  bgfx::TextureRegion a,b;a.handle=image.value;b.handle=staging.value;bgfx::blit(7,b,a);std::vector<std::uint32_t> result(256*height*4);const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
 };
 const auto grain=[&](unsigned slot,float x,float y,float z,unsigned material){return ParticleGpu{{x,y,z,.0625f},{0,0,0,terrain::conversionMassProfile(materials[material].density).inverseMass},{293,0,0,0},{material,1,slot,1}};};
 unsigned cases=0,totalClaims=0;
 const auto run=[&](const std::vector<ParticleGpu>& particles,const std::vector<SettlingState>& history,unsigned quota,double tolerance,unsigned errors=0,bool exact=true){
  const auto cap=unsigned(particles.size());const ParticleGpu zero{};const SettlingState empty{};
  ClaimHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(cap?particles.data():&zero,std::max(1u,cap)*64),lanes,BGFX_BUFFER_COMPUTE_READ));ClaimHandle histories(bgfx::createDynamicVertexBuffer(bgfx::copy(cap?history.data():&empty,std::max(1u,cap)*16),lanes,BGFX_BUFFER_COMPUTE_READ));
  GpuDepositClaims claims(assets,cap,materials,quota,tolerance);PoolDiagnostics diagnostic(assets);claims.submit(1,input.value,histories.value,occupancy,world);check(diagnostic.request(6,claims.status(),0),"Claim status request failed");const auto output=read(claims.claims(),quota,2);while(diagnostic.pending()){diagnostic.poll(bgfx::frame());SDL_PumpEvents();}const auto status=diagnostic.latest();
  check(status.active==errors && status.free<=quota && status.reserved==status.free*2,"Deposit claim status/counts invalid");
  if(errors)check(status.free==0,"Faulted input retained deposit claims");
  if(!errors && exact){const auto reference=terrain::planDeposits(world,particles,history,materials,0,quota,tolerance);if(status.free!=reference.claims.size()){for(unsigned c=0;c<status.free;++c)std::cerr<<"claim "<<c<<" owners "<<output[c*8+4]<<","<<output[c*8+6]<<" cell "<<std::bit_cast<std::int32_t>(output[c*8])<<","<<std::bit_cast<std::int32_t>(output[c*8+2])<<"\n";throw std::runtime_error("GPU/CPU deposit count differs: case "+std::to_string(cases)+" capacity "+std::to_string(cap)+" GPU "+std::to_string(status.free)+" CPU "+std::to_string(reference.claims.size())+" deferred "+std::to_string(status.rejected));}
   for(unsigned c=0;c<reference.claims.size();++c){const auto& expected=reference.claims[c];const auto cell=*terrain::voxelOf(expected.edit.address);for(unsigned a=0;a<3;++a)check(std::bit_cast<std::int32_t>(output[c*8+a])==cell[a],"GPU deposit target differs from CPU");check(output[c*8+3]==expected.edit.material && output[c*8+4]==expected.grains[0].slot && output[c*8+5]==expected.grains[0].generation && output[c*8+6]==expected.grains[1].slot && output[c*8+7]==expected.grains[1].generation,"GPU deposit owners differ from CPU");}
  }
  std::set<unsigned> retired;std::set<terrain::GlobalVoxel> cells;const auto ownership=readOwners(claims,cap);std::vector<unsigned> expectedOwners(cap);
  for(unsigned c=0;c<status.free;++c){terrain::GlobalVoxel cell{};for(unsigned a=0;a<3;++a)cell[a]=std::bit_cast<std::int32_t>(output[c*8+a]);check(cells.insert(cell).second && world.materialAt(*terrain::addressOf(cell))==terrain::SparseVoxelStore::empty,"Duplicate/occupied claim target");
   const auto material=output[c*8+3];check(material<materials.size(),"Invalid claim material");long double sourceMass=0;
   for(unsigned lane:{4u,6u}){const auto slot=output[c*8+lane];check(slot<cap && retired.insert(slot).second && particles[slot].identity[3]==output[c*8+lane+1] && particles[slot].identity[0]==material,"Duplicate/stale/mixed claim owner");sourceMass+=1.L/particles[slot].velocityInvMass[3];expectedOwners[slot]=c+1;}
   const long double destination=static_cast<long double>(materials[material].density)/512;check(std::abs(destination-sourceMass)<=tolerance*sourceMass,"Deposit mass exceeds tolerance");
  }
  for(unsigned i=0;i<cap;++i)check(ownership[i*4]==expectedOwners[i],"Private retirement ownership differs from claims");
  for(const auto& cell:cells)for(unsigned i=0;i<cap;++i)if(particles[i].identity[1] && !retired.contains(i)){long double squared=0;for(unsigned a=0;a<3;++a){const long double low=static_cast<long double>(cell[a])/8,high=low+.125L,position=particles[i].positionRadius[a];const auto closest=std::clamp(position,low,high);squared+=(position-closest)*(position-closest);}check(squared>=static_cast<long double>(particles[i].positionRadius[3])*particles[i].positionRadius[3],"Accepted deposit penetrates remaining grain");}
  for(unsigned c=status.free;c<quota;++c)check(output[c*8]==UINT32_MAX && output[c*8+3]==UINT32_MAX && output[c*8+4]==UINT32_MAX && output[c*8+6]==UINT32_MAX,"Unused claim was not cleared");
  const auto retiredState=read(claims.retired(),cap,4);
  for(unsigned i=0;i<cap;++i){auto expected=particles[i];if(retired.contains(i))expected.identity[1]=0;check(std::memcmp(retiredState.data()+i*16,&expected,64)==0,"Private retirement changed wrong particle bytes");}
  check(diagnostic.request(6,claims.retirementStatus(),0),"Retirement status request failed");while(diagnostic.pending()){diagnostic.poll(bgfx::frame());SDL_PumpEvents();}const auto retirement=diagnostic.latest();check(retirement.active==errors && retirement.free==status.free && retirement.reserved==status.free*2,"Retirement acknowledgement mismatch");
  const auto after=read(input.value,cap,4),historyAfter=read(histories.value,cap,1);if(cap)check(std::memcmp(after.data(),particles.data(),cap*64)==0 && std::memcmp(historyAfter.data(),history.data(),cap*16)==0,"Claim preparation mutated live source/history");
  if(cap==2 && status.free==1 && errors==0){
   ClaimHandle retireProgram(loadCompute(assets,"cs_deposit_retire"));ClaimHandle retireParams(bgfx::createUniform("u_deposit_claims",bgfx::UniformType::Vec4));
   for(unsigned fault=0;fault<7;++fault){
    claims.submit(1,input.value,histories.value,occupancy,world);bgfx::frame();
    auto bad=output;std::array<unsigned,2> owners{1,1};std::array<unsigned,4> counts{0,1,2,0};unsigned expectedError=64;
    if(fault==0)bad[5]+=1; // adjacent stale generation, exact uint comparison
    if(fault==1)bad[6]=bad[4]; // duplicate grain
    if(fault==2)bad[6]=cap; // out of bounds; must never dereference
    if(fault==3)owners[1]=0;
    if(fault==4)owners[1]=2;
    if(fault==5)counts[1]=quota+1;
    if(fault==6)counts[0]=32; // upstream health fault must roll back
    if(fault==0)expectedError=128;if(fault==6)expectedError=32;
    bgfx::update(claims.claims(),0,bgfx::copy(bad.data(),unsigned(bad.size()*4)));
    bgfx::update(claims.owners(),0,bgfx::copy(owners.data(),8));
    bgfx::update(claims.status(),0,bgfx::copy(counts.data(),16));
    const float p[]{float(cap),float(quota),0,0};bgfx::setUniform(retireParams.value,p);
    bgfx::setBuffer(0,input.value,bgfx::Access::Read);bgfx::setBuffer(1,claims.claims(),bgfx::Access::Read);bgfx::setBuffer(2,claims.owners(),bgfx::Access::Read);bgfx::setBuffer(3,claims.status(),bgfx::Access::Read);bgfx::setBuffer(4,claims.retired(),bgfx::Access::Write);bgfx::setBuffer(5,claims.retirementStatus(),bgfx::Access::Write);bgfx::dispatch(1,retireProgram.value,1);
    check(diagnostic.request(6,claims.retirementStatus(),0),"Faulted retirement status request failed");const auto rollback=read(claims.retired(),cap,4);while(diagnostic.pending()){diagnostic.poll(bgfx::frame());SDL_PumpEvents();}
    const auto ack=diagnostic.latest();check(ack.active==expectedError && ack.free==0 && ack.reserved==0 && std::memcmp(rollback.data(),particles.data(),cap*64)==0,"Faulted retirement did not preserve whole source");
   }
  }
  claims.submit(1,input.value,histories.value,occupancy,world);const auto repeat=read(claims.claims(),quota,2);if(quota)check(output==repeat,"Repeated GPU deposit claims changed");
  rejects([&]{claims.submit(1,claims.claims(),histories.value,occupancy,world);});rejects([&]{claims.submit(1,claims.retired(),histories.value,occupancy,world);});totalClaims+=status.free;++cases;return status.free;
 };
 for(unsigned cap:{0u,1u,2u,127u,128u,129u,255u,256u,257u,1024u,4096u})for(unsigned scenario=0;scenario<5;++scenario){
  std::vector<ParticleGpu> input(cap);std::vector<SettlingState> history(cap);
  for(unsigned slot=0;slot<cap;slot+=3){const unsigned group=slot/3,material=group%2;const float x=-8+float(group%32)*.5f,z=-8+float(group/32)*.5f;
   for(unsigned k=0;k<2 && slot+k<cap;++k){input[slot+k]=grain(slot+k,x+.0625f+float(k)*.125f,.0625f,z+.0625f,material);history[slot+k]={1,30,0,1};}
   if(scenario==1 && slot+2<cap && group%2==0)input[slot+2]=grain(slot+2,x+.125f,.15f,z+.0625f,material);
  }
  if(scenario==2 && cap)input[0].identity[3]=0;
  run(input,history,scenario==3?0:512,scenario==4?.01:.05,scenario==2 && cap?1:0);
 }
 std::mt19937 random(34921);
 for(unsigned scenario=0;scenario<32;++scenario){std::vector<ParticleGpu> input;std::vector<SettlingState> history;
  for(unsigned group=0;group<32;++group){const float x=-4+float(group%8)*.5f,z=-4+float(group/8)*.5f,jitter=float(random()%1000)*.00001f;const unsigned material=random()%2;
   for(unsigned k=0;k<2;++k){input.push_back(grain(unsigned(input.size()),x+.0625f+float(k)*.125f+jitter,.0625f,z+.0625f,material));history.push_back({1,30,0,1});}
   if(random()%3==0){input.push_back(grain(unsigned(input.size()),x+.125f,.15f,z+.0625f,material));history.push_back({});}
  }
  run(input,history,random()%33,.05);
 }
 for(unsigned scenario=0;scenario<4;++scenario){std::vector<ParticleGpu> input{grain(0,.0625f,.0625f,.0625f,0),grain(1,.1875f,.0625f,.0625f,0),grain(2,.125f,.1875f,.0625f,0)};std::vector<SettlingState> history{{1,30,0,1},{1,30,0,1},{}};
  if(scenario==1)input[2].positionRadius[1]=std::nextafter(.1875f,0.f);
  if(scenario==2){input[2]=grain(2,.0625f,.0625f,.1875f,0);history[2]={1,30,0,1};}
  if(scenario==3){input[2].positionRadius[0]=input[2].positionRadius[1]=std::nextafter(float(.125+.0625/std::sqrt(2.)),1.f);}
  const auto count=run(input,history,32,.05,0,scenario!=3);check(count==(scenario==1 || scenario==3?0u:1u),"Boundary/tie deposit case unexpected");
 }
 for(unsigned generation:{16777216u,2147483648u,UINT32_MAX-1}){
  std::vector<ParticleGpu> input{grain(0,.0625f,.0625f,.0625f,0),grain(1,.1875f,.0625f,.0625f,0)};for(auto& p:input)p.identity[3]=generation;std::vector<SettlingState> history(2,SettlingState{generation,30,0,1});run(input,history,32,.05);
 }
 check(world.revision()==0 && world.occupiedVoxelCount()==0,"GPU claims mutated terrain");
 std::cout<<"GPU deposit claims passed: "<<cases<<" batches / "<<totalClaims<<" claims; CPU parity, unique targets/owners, conservative clearance, mass, repeatability and immutable source/history\n";
}
}
