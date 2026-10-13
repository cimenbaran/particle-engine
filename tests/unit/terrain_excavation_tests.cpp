#include "core/TerrainExcavation.hpp"
#include <iostream>
#include <random>
#define CHECK(v) do{if(!(v))throw std::runtime_error("Excavation check failed: " #v);}while(false)
using namespace pw;using namespace pw::terrain;
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}CHECK(bad);}
int main() {
    const std::array<ExcavationMaterial,3> materials{{{2600,true,0},{1600,true,0},{600,true,1}}};
    unsigned cases=0;
    auto command=[] {Command c;c.type=CommandType::EditTerrain;c.tick=13;c.sequence=7;c.minimum={0,0,0};c.maximum={3.9f,3.9f,3.9f};c.count=32;c.parameters[1]=2;return c;};
    auto verify=[&](const SparseVoxelStore& store,const Command& c,unsigned cap) {
        const auto revision=store.revision(),original=store.occupiedVoxelCount();const auto plan=planExcavation(store,c,13,materials,cap),again=planExcavation(store,c,13,materials,cap);
        CHECK(store.revision()==revision && store.occupiedVoxelCount()==original && plan.revision==revision && plan.tick==13 && plan.sequence==7);
        CHECK(plan.state==again.state && plan.removals.size()==again.removals.size() && plan.seeds.size()==again.seeds.size());
        if(plan.state!=ExcavationPlan::State::Ready){CHECK(plan.removals.empty() && plan.seeds.empty());++cases;return plan;}
        CHECK(plan.removals.size()<=c.count && plan.seeds.size()<=cap && plan.eligible>=plan.removals.size());
        auto changed=store.clone();CHECK(changed->setMaterials(plan.removals,revision)==SparseVoxelStore::Edit::Changed);
        std::array<unsigned,3> removed{},grains{};std::set<std::array<float,3>> unique;
        for(unsigned i=0;i<plan.removals.size();++i){CHECK(plan.removals[i].address==again.removals[i].address);++removed[*store.materialAt(plan.removals[i].address)];}
        for(unsigned i=0;i<plan.seeds.size();++i) {
            const auto& s=plan.seeds[i];CHECK(s.position==again.seeds[i].position && s.material==again.seeds[i].material && s.velocity==again.seeds[i].velocity);++grains[s.material];
            CHECK(unique.insert(s.position).second && s.fuel==materials[s.material].fuel && s.temperature==293.15f && s.velocity[1]==2);
            bool local=false;for(const auto& edit:plan.removals)if(*store.materialAt(edit.address)==s.material){const auto corner=*relativeCorner(edit.address,{});bool near=true;for(unsigned a=0;a<3;++a)near&=std::abs(double(s.position[a])-corner[a]-.0625)<=.5;local|=near;}CHECK(local);
            const auto address=addressAtRelative({s.position[0],s.position[1],s.position[2]});CHECK(address && changed->materialAt(*address)==SparseVoxelStore::empty);
            for(unsigned j=0;j<i;++j){double squared=0;for(unsigned a=0;a<3;++a){const double d=double(s.position[a])-plan.seeds[j].position[a];squared+=d*d;}CHECK(squared>=.015625);}
            // Independent point-to-box distance against every nearby remaining
            // voxel, not the planner's centre-cell occupancy shortcut.
            const auto global=*voxelOf(*address);
            for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                const auto other=*addressOf({global[0]+dx,global[1]+dy,global[2]+dz});const auto m=changed->materialAt(other);if(!m || *m==SparseVoxelStore::empty)continue;
                const auto corner=*relativeCorner(other,{});double squared=0;for(unsigned a=0;a<3;++a){const double delta=std::max({corner[a]-s.position[a],double(s.position[a])-corner[a]-.125,0.});squared+=delta*delta;}CHECK(squared>=.00390625);
            }
        }
        for(unsigned m=0;m<3;++m){const auto mass=planVoxelDebris(removed[m],materials[m].density);CHECK(mass.admissible && grains[m]==mass.grains);}
        ++cases;return plan;
    };
    SparseVoxelStore store(ResidentChunkLayout({-1,0,-1},{2,1,2}),3);auto c=command();CHECK(verify(store,c,64).state==ExcavationPlan::State::Empty);
    const auto voxel=*addressOf({8,8,8});store.setMaterial(voxel,0,0);CHECK(verify(store,c,0).state==ExcavationPlan::State::NoCapacity);CHECK(verify(store,c,1).state==ExcavationPlan::State::NoCapacity);
    CHECK(verify(store,c,2).seeds.size()==2);
    c.minimum=c.maximum={1.0625f,1.0625f,1.0625f};CHECK(verify(store,c,2).eligible==1);c.minimum[0]=std::nextafter(c.minimum[0],2.f);c.maximum[0]=c.minimum[0];CHECK(verify(store,c,2).state==ExcavationPlan::State::Empty);
    c=command();std::mt19937 random(99034);
    for(unsigned trial=0;trial<128;++trial) {
        SparseVoxelStore randomStore(ResidentChunkLayout({-1,0,-1},{2,1,2}),3);std::vector<SparseVoxelStore::VoxelEdit> edits;
        for(int y=0;y<8;++y)for(int z=-4;z<4;++z)for(int x=-4;x<4;++x)if(random()%3)edits.push_back({*addressOf({x,y,z}),random()%3});randomStore.setMaterials(edits,0);
        c.minimum={-.5f,0,-.5f};c.maximum={.5f,1,.5f};c.count=1+random()%128;
        const auto p=verify(randomStore,c,2+random()%256);CHECK(p.eligible==edits.size());
    }
    SparseVoxelStore dense(ResidentChunkLayout({0,0,0},{1,1,1}),3);std::vector<SparseVoxelStore::VoxelEdit> edits;
    for(int y=0;y<32;++y)for(int z=0;z<32;++z)for(int x=0;x<32;++x)edits.push_back({*addressOf({x,y,z}),0});dense.setMaterials(edits,0);
    c=command();c.minimum=c.maximum={2.0625f,2.0625f,2.0625f};CHECK(verify(dense,c,64).state==ExcavationPlan::State::NoSpace);
    // Resident edge: no candidate may spill into a nonresident cell.
    c.minimum=c.maximum={.0625f,.0625f,.0625f};CHECK(verify(dense,c,64).state==ExcavationPlan::State::NoSpace);
    c=command();c.count=512;CHECK(verify(store,c,4096).state==ExcavationPlan::State::Ready);
    SparseVoxelStore wide(ResidentChunkLayout({-4,0,-1},{8,1,2}),3);wide.setMaterial(*addressOf({-120,8,0}),0,0);wide.setMaterial(*addressOf({120,8,0}),1,1);
    c=command();c.minimum={-16,1,-.5f};c.maximum={16,1.2f,.5f};CHECK(verify(wide,c,4).seeds.size()==4);
    for(unsigned fault=0;fault<10;++fault) {
        auto invalid=command();unsigned cap=64;
        if(fault==0)invalid.tick=12;if(fault==1)invalid.tick=14;if(fault==2)invalid.type=CommandType::SpawnBrush;if(fault==3)invalid.count=0;if(fault==4)invalid.count=513;
        if(fault==5)invalid.parameters[0]=13;if(fault==6)invalid.parameters[3]=1;if(fault==7)invalid.minimum[0]=std::numeric_limits<float>::quiet_NaN();if(fault==8)invalid.sequence=UINT64_MAX;if(fault==9)cap=4097;
        rejects([&]{planExcavation(store,invalid,13,materials,cap);});
    }
    c=command();c.minimum={-16,0,-16};c.maximum={16,16,16};rejects([&]{planExcavation(store,c,13,materials,64);});
    auto bad=materials;bad[0].density=0;c=command();rejects([&]{planExcavation(store,c,13,bad,64);});bad=materials;bad[0].solid=false;rejects([&]{planExcavation(store,c,13,bad,64);});
    std::cout<<"Terrain excavation passed: "<<cases<<" deterministic plans, independent clearance/mass oracles, quotas, bounds, residency and rejection\n";
}
