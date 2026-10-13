#include "core/TerrainExplosion.hpp"
#include "core/BasinWorld.hpp"
#include <iostream>
#include <random>
#include <set>
#define CHECK(v) do{if(!(v))throw std::runtime_error("Explosion check failed: " #v);}while(false)
using namespace pw::terrain;
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}CHECK(bad);}
int main() {
    const std::array<ExplosionMaterial,3> materials{{{2600,true,1000},{1600,true,100},{600,true,300}}};
    ExplosionRequest request;request.tick=3;request.sequence=9;request.centre={1.0625f,1.0625f,1.0625f};request.radius=.5f;request.energyJoules=100000;
    unsigned plans=0;
    auto verify=[&](const SparseVoxelStore& store,const ExplosionRequest& r) {
        const auto revision=store.revision(),count=store.occupiedVoxelCount();
        const auto plan=planExplosion(store,r,r.tick,materials),repeat=planExplosion(store,r,r.tick,materials);
        CHECK(store.revision()==revision && store.occupiedVoxelCount()==count);
        CHECK(plan.revision==revision && plan.tick==r.tick && plan.sequence==r.sequence);
        CHECK(plan.removals.size()==repeat.removals.size() && plan.reservedJoules==repeat.reservedJoules);
        CHECK(plan.removals.size()<=r.voxelQuota && plan.grains<=r.grainQuota && plan.reservedJoules<=r.energyJoules);
        CHECK(plan.removals.size()+plan.capacityDeferred+plan.energyRejected+plan.placementDeferred==plan.eligible);
        std::array<unsigned,3> voxels{};std::set<std::array<std::int64_t,3>> unique;double fracture=0,kinetic=0;
        for(unsigned i=0;i<plan.removals.size();++i) {
            const auto& edit=plan.removals[i];CHECK(edit.material==SparseVoxelStore::empty && edit.address==repeat.removals[i].address);
            const auto global=*voxelOf(edit.address);CHECK(unique.insert(global).second);
            const auto m=*store.materialAt(edit.address);CHECK(m<materials.size());++voxels[m];
            const auto corner=*relativeCorner(edit.address,{});double squared=0;
            for(unsigned a=0;a<3;++a){const double delta=corner[a]+.0625-r.centre[a];squared+=delta*delta;}
            CHECK(squared<double(r.radius)*r.radius);const double linear=1-std::sqrt(squared)/r.radius;
            fracture+=materials[m].resistance/512/(linear*linear);
        }
        unsigned grains=0;
        for(unsigned m=0;m<materials.size();++m){const auto balance=planVoxelDebris(voxels[m],materials[m].density);CHECK(balance.admissible && balance.grains==plan.grainsPerMaterial[m]);grains+=balance.grains;kinetic+=.5*balance.destinationKilograms*r.impulse*r.impulse;}
        CHECK(grains==plan.grains && std::abs(plan.fractureJoules-fracture)<=1e-10*std::max(1.,fracture));
        CHECK(std::abs(plan.kineticJoules-kinetic)<=1e-10*std::max(1.,kinetic));
        CHECK(plan.reservedJoules+1e-12*std::max(1.,plan.reservedJoules)>=fracture+kinetic);
        CHECK(plan.seeds.size()==plan.grains);double actualKinetic=0;
        for(const auto& seed:plan.seeds) {
            const double speed=std::hypot(double(seed.velocity[0]),double(seed.velocity[1]),double(seed.velocity[2]));
            CHECK(speed<=r.impulse);actualKinetic+=.5*conversionMassProfile(materials[seed.material].density).kilograms*speed*speed;
            std::array<double,3> delta;for(unsigned a=0;a<3;++a)delta[a]=double(seed.position[a])-r.centre[a];
            const double distance=std::hypot(delta[0],delta[1],delta[2]);
            if(distance>0)for(unsigned a=0;a<3;++a)CHECK(seed.velocity[a]*delta[a]>=0);
            const double linear=std::max(0.,1-distance/r.radius);CHECK(speed<=r.impulse*linear*linear+1e-12);
        }
        CHECK(actualKinetic<=plan.kineticJoules+1e-12*std::max(1.,actualKinetic));
        ++plans;return plan;
    };
    SparseVoxelStore store(ResidentChunkLayout({0,0,0},{1,1,1}),3);
    CHECK(verify(store,request).state==ExplosionPlan::State::Empty);
    const auto centre=*addressOf({8,8,8});store.setMaterial(centre,0,0);
    auto one=verify(store,request);CHECK(one.removals.size()==1 && one.grains==2);
    auto low=request;low.energyJoules=one.reservedJoules;CHECK(verify(store,low).removals.size()==1);
    low.energyJoules=std::nextafter(one.reservedJoules,0.);CHECK(verify(store,low).state==ExplosionPlan::State::InsufficientEnergy);
    low=request;low.grainQuota=1;CHECK(verify(store,low).state==ExplosionPlan::State::NoCapacity);
    low.grainQuota=0;CHECK(verify(store,low).removals.empty());
    low=request;low.energyJoules=0;CHECK(verify(store,low).state==ExplosionPlan::State::InsufficientEnergy);
    low=request;low.impulse=0;CHECK(verify(store,low).kineticJoules==0);
    store.setMaterial(*addressOf({12,8,8}),0,store.revision());CHECK(verify(store,request).eligible==1); // exact radius excluded
    store.setMaterial(*addressOf({10,8,8}),1,store.revision());CHECK(verify(store,request).eligible==2);
    auto weaker=request;weaker.impulse=0;weaker.energyJoules=1;
    auto selected=verify(store,weaker);CHECK(selected.removals.size()==1 && *store.materialAt(selected.removals[0].address)==1); // strong centre retained, weak outer cell fits
    auto tight=request;tight.voxelQuota=1;CHECK(verify(store,tight).removals[0].address==centre);
    // Random mixed-material sparse worlds exercise independent radial/cost/mass
    // checks at varied budgets and both caps without mutating their source.
    std::mt19937 random(77019);
    for(unsigned trial=0;trial<128;++trial) {
        SparseVoxelStore source(ResidentChunkLayout({-1,0,-1},{2,1,2}),3);std::vector<SparseVoxelStore::VoxelEdit> edits;
        for(int z=-6;z<6;++z)for(int y=0;y<10;++y)for(int x=-6;x<6;++x)if(random()%4==0)edits.push_back({*addressOf({x,y,z}),random()%3});
        source.setMaterials(edits,0);auto r=request;r.centre={0,.5f,0};r.radius=.125f+float(random()%16)/16;r.impulse=float(random()%13);r.energyJoules=random()%10000;r.voxelQuota=1+random()%128;r.grainQuota=random()%257;
        verify(source,r);
    }
    // Simulate acknowledgement by applying only complete proposals. Every
    // unconverted voxel remains available to a later batch with remaining energy.
    SparseVoxelStore staged(ResidentChunkLayout({0,0,0},{1,1,1}),3);std::vector<SparseVoxelStore::VoxelEdit> edits;
    for(int z=5;z<12;++z)for(int x=5;x<12;++x)edits.push_back({*addressOf({x,8,z}),0});staged.setMaterials(edits,0);
    auto r=request;r.radius=1;r.voxelQuota=7;double remaining=r.energyJoules;unsigned batches=0,removed=0;
    while(staged.occupiedVoxelCount()) {
        r.energyJoules=remaining;const auto proposal=verify(staged,r);CHECK(proposal.state==ExplosionPlan::State::Ready);
        CHECK(staged.setMaterials(proposal.removals,proposal.revision)==SparseVoxelStore::Edit::Changed);
        remaining=explosionEnergyRemainder(remaining,proposal.reservedJoules);removed+=unsigned(proposal.removals.size());++batches;++r.tick;
    }
    CHECK(batches==7 && removed==49 && remaining>=0);
    CHECK(explosionEnergyRemainder(10,0)==10 && explosionEnergyRemainder(10,10)==0 && explosionEnergyRemainder(10,1)<=9);
    rejects([&]{explosionEnergyRemainder(1,2);});rejects([&]{explosionEnergyRemainder(1,-1);});rejects([&]{explosionEnergyRemainder(INFINITY,0);});
    // Basin edge and sparse residency: never allocate or propose outside it.
    r=request;r.centre={16,0,16};r.radius=4;CHECK(verify(store,r).state==ExplosionPlan::State::Empty);
    auto basin=buildBasin(1337,3,0,1,2);r=request;r.centre={-8,3.875f,-4};
    const auto basinPlan=verify(*basin,r);CHECK(basinPlan.state==ExplosionPlan::State::Ready && basinPlan.removals.size()==32 && basinPlan.seeds.size()==61);
    SparseVoxelStore buried(ResidentChunkLayout({0,0,0},{1,1,1}),3);edits.clear();
    for(int y=0;y<32;++y)for(int z=0;z<32;++z)for(int x=0;x<32;++x)edits.push_back({*addressOf({x,y,z}),0});buried.setMaterials(edits,0);
    r=request;r.centre={2.0625f,2.0625f,2.0625f};r.radius=.01f;
    const auto blocked=verify(buried,r);CHECK(blocked.state==ExplosionPlan::State::NoSpace && blocked.reservedJoules==0 && blocked.removals.empty());
    for(unsigned fault=0;fault<13;++fault) {
        auto bad=request;
        if(fault==0)bad.tick=4;if(fault==1)bad.sequence=UINT64_MAX;if(fault==2)bad.radius=0;if(fault==3)bad.radius=4.01f;if(fault==4)bad.impulse=-1;if(fault==5)bad.impulse=13;
        if(fault==6)bad.energyJoules=-1;if(fault==7)bad.energyJoules=1e9+1;if(fault==8)bad.centre[1]=-1;if(fault==9)bad.centre[0]=NAN;if(fault==10)bad.voxelQuota=513;if(fault==11)bad.voxelQuota=0;if(fault==12)bad.grainQuota=4097;
        rejects([&]{planExplosion(store,bad,3,materials);});
    }
    auto bad=materials;bad[0].resistance=0;rejects([&]{planExplosion(store,request,3,bad);});bad=materials;bad[0].resistance=INFINITY;rejects([&]{planExplosion(store,request,3,bad);});bad=materials;bad[0].solid=false;rejects([&]{planExplosion(store,request,3,bad);});
    std::cout<<"Terrain explosion passed: "<<plans<<" plans, radial/resistance/energy/mass oracles, bounded batches, deterministic retries and immutable source\n";
}
