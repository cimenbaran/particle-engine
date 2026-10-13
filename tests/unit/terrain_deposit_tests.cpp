#include "core/TerrainDeposit.hpp"
#include <cstring>
#include <iostream>
#include <set>
#include <random>
#define CHECK(x) do{if(!(x))throw std::runtime_error("Deposit check failed: " #x);}while(false)
using namespace pw;using namespace pw::terrain;
int main(){try{
 const std::array materials{DepositMaterial{2600,true},DepositMaterial{1600,true},DepositMaterial{1000,false}};
 const auto grain=[&](unsigned slot,float x,float y,float z,unsigned material=0){return ParticleGpu{{x,y,z,.0625f},{0,0,0,conversionMassProfile(materials[material].density).inverseMass},{293.15f,0,0,0},{material,1,slot,1}};};
 SparseVoxelStore world(ResidentChunkLayout({-1,0,-1},{2,1,2}),3);
 std::vector<ParticleGpu> particles{grain(0,.0625f,.0625f,.0625f),grain(1,.1875f,.0625f,.0625f)};
 std::vector<SettlingState> history(2,{1,30,0,1});const auto before=particles;
 auto plan=planDeposits(world,particles,history,materials,7);
 CHECK(plan.tick==7 && plan.revision==0 && plan.claims.size()==1 && plan.claims[0].edit.address==*addressOf({0,0,0}));
 CHECK(plan.claims[0].grains[0]==(ParticleId{0,1}) && plan.claims[0].grains[1]==(ParticleId{1,1}));
 CHECK(plan.claims[0].mass.admissible && plan.claims[0].mass.quantizationKilograms<0 && plan.claims[0].mass.relativeError<.05);
 CHECK(world.occupiedVoxelCount()==0 && world.revision()==0 && std::memcmp(before.data(),particles.data(),128)==0);
 CHECK(planDeposits(world,particles,history,materials,7,0).claims.empty());
 CHECK(planDeposits(world,particles,history,materials,7,32,.01).claims.empty());
 for(unsigned bad=0;bad<8;++bad){auto p=particles;auto h=history;
  switch(bad){case 0:h[1].generation=2;break;case 1:h[1].candidate=0;break;case 2:h[1].quietTicks=29;break;case 3:h[1].unsupportedTicks=1;break;case 4:p[1].velocityInvMass[0]=.10001f;break;case 5:p[1].positionRadius[3]=.05f;break;case 6:p[1].velocityInvMass[3]*=2;break;case 7:p[1]=grain(1,.1875f,.0625f,.0625f,1);break;}
  CHECK(planDeposits(world,p,h,materials,7).claims.empty());
 }
 // A third grain intersecting the new voxel must remain dynamic and block it.
 auto blocked=particles;blocked.push_back(grain(2,.0625f,.15f,.0625f));auto histories=history;histories.push_back({});
 const auto alternate=planDeposits(world,blocked,histories,materials,7);
 CHECK(alternate.claims.size()==1 && alternate.claims[0].edit.address==*addressOf({1,0,0}));
 blocked[2].positionRadius[0]=.125f;CHECK(planDeposits(world,blocked,histories,materials,7).claims.empty());
 // Exact face tangency is allowed; one float inward is not.
 blocked[2]=grain(2,.125f,.1875f,.0625f);CHECK(planDeposits(world,blocked,histories,materials,7).claims.size()==1);
 blocked[2].positionRadius[1]=std::nextafter(.1875f,0.f);CHECK(planDeposits(world,blocked,histories,materials,7).claims.empty());
 // Stable slot tie breaking is independent of source vector addresses.
 auto tied=particles;tied.push_back(grain(2,.0625f,.0625f,.1875f));histories=history;histories.push_back({1,30,0,1});
 const auto tie=planDeposits(world,tied,histories,materials,7);CHECK(tie.claims.size()==1 && tie.claims[0].grains[1].slot==1);
 // Occupied target cannot be overwritten; unsupported targets cannot float.
 world.setMaterial(*addressOf({0,0,0}),1,0);world.setMaterial(*addressOf({1,0,0}),1,1);CHECK(planDeposits(world,particles,history,materials,7).claims.empty());
 particles={grain(0,.0625f,.1875f,.0625f),grain(1,.1875f,.1875f,.0625f)};
 CHECK(planDeposits(world,particles,history,materials,7).claims.size()==1);
 particles={grain(0,.0625f,.3125f,.0625f),grain(1,.1875f,.3125f,.0625f)};
 CHECK(planDeposits(world,particles,history,materials,7).claims.empty());
 // Signed cells and many independent material claims; no shared owner or target.
 SparseVoxelStore empty(ResidentChunkLayout({-1,0,-1},{2,1,2}),3);particles.clear();history.clear();
 for(unsigned pair=0;pair<64;++pair){float x=-3.5f+float(pair%8)*.5f,z=-3.5f+float(pair/8)*.5f;for(unsigned k=0;k<2;++k){particles.push_back(grain(unsigned(particles.size()),x+.0625f+float(k)*.125f,.0625f,z+.0625f,pair%2));history.push_back({1,30,0,1});}}
 const auto many=planDeposits(empty,particles,history,materials,9,512);CHECK(many.claims.size()==64);CHECK(planDeposits(empty,particles,history,materials,9,17).claims.size()==17);
 std::set<unsigned> retired;std::set<GlobalVoxel> targets;
 for(const auto& claim:many.claims){CHECK(targets.insert(*voxelOf(claim.edit.address)).second);for(auto id:claim.grains)CHECK(retired.insert(id.slot).second);const double mass=2./conversionMassProfile(materials[claim.edit.material].density).inverseMass;CHECK(claim.mass.sourceKilograms==mass && std::abs(claim.mass.destinationKilograms-mass)<=mass*.05);}
 std::mt19937 rng(41892);unsigned randomClaims=0;
 for(unsigned run=0;run<128;++run){
  std::vector<ParticleGpu> input;std::vector<SettlingState> histories;
  for(unsigned pair=0;pair<32;++pair){const float x=-3.5f+float(pair%8)*.5f,z=-3.5f+float(pair/8)*.5f;const unsigned material=rng()%2;
   const float jitter=float(rng()%1000)*.00001f;
   for(unsigned k=0;k<2;++k){input.push_back(grain(unsigned(input.size()),x+.0625f+float(k)*.125f+jitter,.0625f,z+.0625f,material));histories.push_back({1,30,0,1});}
   if(rng()%3==0){input.push_back(grain(unsigned(input.size()),x+.125f,.15f,z+.0625f,material));histories.push_back({});}
  }
  const auto copy=input;const auto historyCopy=histories;const unsigned quota=rng()%33;
  const auto result=planDeposits(empty,input,histories,materials,run,quota);
  const auto repeat=planDeposits(empty,input,histories,materials,run,quota);
  CHECK(result.claims.size()<=quota && result.claims.size()==repeat.claims.size());
  std::set<unsigned> owners;std::set<GlobalVoxel> cells;
  for(unsigned c=0;c<result.claims.size();++c){const auto& claim=result.claims[c];CHECK(claim.edit.address==repeat.claims[c].edit.address && claim.grains==repeat.claims[c].grains);
   const auto cell=*voxelOf(claim.edit.address);CHECK(cells.insert(cell).second && *empty.materialAt(claim.edit.address)==SparseVoxelStore::empty);
   long double sourceMass=0;for(const auto id:claim.grains){CHECK(owners.insert(id.slot).second && input[id.slot].identity[3]==id.generation && input[id.slot].identity[0]==claim.edit.material);sourceMass+=1.L/input[id.slot].velocityInvMass[3];}
   const long double destination=static_cast<long double>(materials[claim.edit.material].density)/512;
   CHECK(std::abs(destination-sourceMass)<=.05L*sourceMass);++randomClaims;
  }
  for(const auto& claim:result.claims){const auto cell=*voxelOf(claim.edit.address);
   for(unsigned id=0;id<input.size();++id)if(!owners.contains(id)){long double squared=0;
    for(unsigned a=0;a<3;++a){const long double low=static_cast<long double>(cell[a])/8,high=low+.125L,position=input[id].positionRadius[a];const auto closest=std::clamp(position,low,high);squared+=(position-closest)*(position-closest);}
    CHECK(squared>=static_cast<long double>(input[id].positionRadius[3])*input[id].positionRadius[3]);
   }
  }
  CHECK(std::memcmp(input.data(),copy.data(),input.size()*sizeof(ParticleGpu))==0 && histories==historyCopy);
 }
 CHECK(randomClaims>1000);
 const auto rejects=[](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}CHECK(bad);};
 rejects([&]{planDeposits(empty,particles,history,materials,9,513);});auto wrong=particles;wrong[0].identity[2]=100;rejects([&]{planDeposits(empty,wrong,history,materials,9);});wrong=particles;wrong[0].positionRadius[0]=NAN;rejects([&]{planDeposits(empty,wrong,history,materials,9);});
 CHECK(empty.occupiedVoxelCount()==0 && empty.revision()==0);
 std::cout<<"Deposit reference passed: "<<randomClaims<<" random claims, mass pairing, support, stale histories, exact tangency, stable conflicts, signed cells, quotas, unique ownership and immutable source\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
