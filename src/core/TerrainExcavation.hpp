#pragma once
#include "Commands.hpp"
#include "TerrainDebrisPlacement.hpp"
#include "SparseVoxelStore.hpp"
#include "TerrainMass.hpp"
#include <set>
#include <span>
namespace pw::terrain {
using ExcavationMaterial=DebrisMaterial;
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
    for(const auto& cell:selected)plan.removals.push_back({*addressOf({cell.x,cell.y,cell.z}),SparseVoxelStore::empty});
    auto placed=placeTerrainDebris(source,plan.removals,materials,{command.parameters[0],command.parameters[1],command.parameters[2]});
    if(!placed){plan.state=ExcavationPlan::State::NoSpace;plan.removals.clear();return plan;}
    plan.seeds=std::move(*placed);
    plan.state=ExcavationPlan::State::Ready;return plan;
}
}
