#pragma once
#include "Commands.hpp"
#include "DebrisSeed.hpp"
#include "SparseVoxelStore.hpp"
#include "TerrainMass.hpp"
#include <set>
#include <span>
namespace pw::terrain {
struct ExcavationMaterial {float density=1000;bool solid=true;float fuel=0;};
struct ExcavationPlan {
    enum class State {Ready,Empty,NoCapacity,NoSpace};
    State state=State::Empty;
    std::uint64_t revision=0,tick=0,sequence=0;
    std::uint32_t eligible=0;
    std::vector<SparseVoxelStore::VoxelEdit> removals;
    std::vector<DebrisSeed> seeds;
};
// CPU proposal only. Voxel centres form a .125m lattice: empty cells give exact
// terrain clearance for .0625m grains. Live-particle clearance is a GPU gate.
// Zero-origin Basin bounds; limited CPU scan and placement envelope.
inline ExcavationPlan planExcavation(const SparseVoxelStore& source,const Command& command,
    std::uint64_t tick,std::span<const ExcavationMaterial> materials,std::uint32_t maxSeeds) {
    CommandLimits limits;limits.worldBound=16;limits.materialCount=std::uint32_t(materials.size());limits.spawnPerCommand=512;
    if(materials.empty() || materials.size()>65536 || command.type!=CommandType::EditTerrain || command.tick!=tick ||
        command.sequence==UINT64_MAX || command.count==0 || maxSeeds>4096 || !validate(command,limits,tick).empty() ||
        std::hypot(double(command.parameters[0]),double(command.parameters[1]),double(command.parameters[2]))>12 || command.parameters[3]!=0)
        throw std::invalid_argument("Invalid excavation command, tick or quota");
    for(const auto& material:materials){conversionMassProfile(material.density);if(!std::isfinite(material.fuel) || material.fuel<0 || material.fuel>1)throw std::invalid_argument("Invalid excavation material");}
    ExcavationPlan plan;plan.revision=source.revision();plan.tick=tick;plan.sequence=command.sequence;
    std::array<int,3> lo,hi;std::uint64_t volume=1;
    for(unsigned a=0;a<3;++a){lo[a]=std::max(a==1?0:-128,int(std::ceil(double(command.minimum[a])*8-.5)));hi[a]=std::min(127,int(std::floor(double(command.maximum[a])*8-.5)));if(lo[a]>hi[a])return plan;volume*=unsigned(hi[a]-lo[a]+1);}
    if(volume>262144)throw std::invalid_argument("Excavation scan exceeds 262144 voxels");
    struct Cell {int x,y,z;std::uint32_t material;};std::vector<Cell> selected;
    std::vector<std::uint32_t> counts(materials.size());std::uint32_t products=0;bool full=false;
    // Surface-first stable order: descending Y, ascending Z/X. Count all
    // eligible voxels even after the batch is full; never hide truncation.
    for(int y=hi[1];y>=lo[1];--y)for(int z=lo[2];z<=hi[2];++z)for(int x=lo[0];x<=hi[0];++x) {
        const auto address=*addressOf({x,y,z});const auto material=source.materialAt(address);
        if(!material || *material==SparseVoxelStore::empty)continue;
        if(*material>=materials.size() || !materials[*material].solid)throw std::invalid_argument("Excavation requires solid resident terrain");
        ++plan.eligible;if(full)continue;
        const auto previous=planVoxelDebris(counts[*material],materials[*material].density);
        const auto next=planVoxelDebris(counts[*material]+1,materials[*material].density);
        if(!next.admissible || products+next.grains-previous.grains>maxSeeds || selected.size()>=command.count){full=true;continue;}
        products+=next.grains-previous.grains;++counts[*material];selected.push_back({x,y,z,*material});
    }
    if(!plan.eligible)return plan;
    if(selected.empty()){plan.state=ExcavationPlan::State::NoCapacity;return plan;}
    auto key=[](int x,int y,int z){return std::uint32_t(x+128+256*(y+128*(z+128)));};
    std::set<std::uint32_t> occupied;
    auto seed=[&](int x,int y,int z,unsigned material) {
        DebrisSeed result;result.position={(float(x)+.5f)*.125f,(float(y)+.5f)*.125f,(float(z)+.5f)*.125f};
        result.velocity={command.parameters[0],command.parameters[1],command.parameters[2]};result.material=material;result.fuel=materials[material].fuel;
        plan.seeds.push_back(result);
    };
    // Preserve one product at each removed voxel centre, then place the extra
    // mass-accounted products in nearby empty cells (within a .5m envelope).
    for(const auto& cell:selected) {
        plan.removals.push_back({*addressOf({cell.x,cell.y,cell.z}),SparseVoxelStore::empty});
        seed(cell.x,cell.y,cell.z,cell.material);occupied.insert(key(cell.x,cell.y,cell.z));
    }
    struct Site {int x,y,z,distance;};std::map<unsigned,std::vector<Site>> sites;
    // Enumerate only the union of .5m neighborhoods of removed voxels, grouped
    // by material. A wide sparse brush must not relocate debris to its midpoint.
    // At most 512 * 9^3 candidate checks; duplicates are filtered during placement.
    for(const auto& cell:selected)for(int dy=-4;dy<=4;++dy)for(int dz=-4;dz<=4;++dz)for(int dx=-4;dx<=4;++dx) {
        const int x=cell.x+dx,y=cell.y+dy,z=cell.z+dz;
        if(x<-128 || x>127 || y<0 || y>127 || z<-128 || z>127 || occupied.contains(key(x,y,z)))continue;
        const auto material=source.materialAt(*addressOf({x,y,z}));if(!material || *material!=SparseVoxelStore::empty)continue;
        sites[cell.material].push_back({x,y,z,dx*dx+dy*dy+dz*dz});
    }
    for(unsigned material=0;material<counts.size();++material) {
        if(!counts[material])continue;
        auto& candidates=sites[material];
        std::sort(candidates.begin(),candidates.end(),[](const Site& a,const Site& b){if(a.distance!=b.distance)return a.distance<b.distance;if(a.y!=b.y)return a.y>b.y;if(a.z!=b.z)return a.z<b.z;return a.x<b.x;});
        const auto total=planVoxelDebris(counts[material],materials[material].density).grains;unsigned placed=counts[material];
        for(const auto& p:candidates) {
            if(placed==total)break;
            if(!occupied.insert(key(p.x,p.y,p.z)).second)continue;
            seed(p.x,p.y,p.z,material);++placed;
        }
        if(placed!=total){plan.state=ExcavationPlan::State::NoSpace;plan.removals.clear();plan.seeds.clear();return plan;}
    }
    plan.state=ExcavationPlan::State::Ready;return plan;
}
}
