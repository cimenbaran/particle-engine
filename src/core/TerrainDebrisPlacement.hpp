#pragma once
#include "DebrisSeed.hpp"
#include "SparseVoxelStore.hpp"
#include "TerrainMass.hpp"
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <span>
namespace pw::terrain {
struct DebrisMaterial {float density=1000;bool solid=true;float fuel=0;};
// Bounded local placement for an already selected removal batch. No mutation.
// A missing result means the complete mass-accounted products cannot fit.
inline std::optional<std::vector<DebrisSeed>> placeTerrainDebris(const SparseVoxelStore& source,
    std::span<const SparseVoxelStore::VoxelEdit> removals,std::span<const DebrisMaterial> materials,
    std::array<float,3> velocity={}) {
    if(removals.empty() || removals.size()>512 || materials.empty() || materials.size()>65536)
        throw std::invalid_argument("Invalid terrain debris placement batch");
    for(float v:velocity)if(!std::isfinite(v))throw std::invalid_argument("Invalid debris velocity");
    if(std::hypot(double(velocity[0]),double(velocity[1]),double(velocity[2]))>12)throw std::invalid_argument("Excessive debris velocity");
    for(const auto& m:materials){conversionMassProfile(m.density);if(!std::isfinite(m.fuel) || m.fuel<0 || m.fuel>1)throw std::invalid_argument("Invalid debris fuel");}
    struct Cell {int x,y,z;unsigned material;};std::vector<Cell> selected;
    std::vector<unsigned> counts(materials.size());std::set<std::array<std::int64_t,3>> unique;
    for(const auto& removal:removals) {
        const auto global=voxelOf(removal.address);const auto material=source.materialAt(removal.address);
        if(removal.material!=SparseVoxelStore::empty || !global || !material || *material==SparseVoxelStore::empty ||
           *material>=materials.size() || !materials[*material].solid || !unique.insert(*global).second ||
           (*global)[0]<-128 || (*global)[0]>127 || (*global)[1]<0 || (*global)[1]>127 || (*global)[2]<-128 || (*global)[2]>127)
            throw std::invalid_argument("Invalid debris removal site");
        selected.push_back({int((*global)[0]),int((*global)[1]),int((*global)[2]),*material});++counts[*material];
    }
    std::vector<DebrisSeed> seeds;
    auto key=[](int x,int y,int z){return std::uint32_t(x+128+256*(y+128*(z+128)));};
    std::set<std::uint32_t> occupied;
    auto seed=[&](int x,int y,int z,unsigned material) {
        DebrisSeed result;result.position={(float(x)+.5f)*.125f,(float(y)+.5f)*.125f,(float(z)+.5f)*.125f};
        result.velocity={velocity[0],velocity[1],velocity[2]};result.material=material;result.fuel=materials[material].fuel;
        seeds.push_back(result);
    };
    // Preserve one product at each removed voxel centre, then place the extra
    // mass-accounted products in nearby empty cells (within a .5m envelope).
    for(const auto& cell:selected) {
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
        if(placed!=total)return std::nullopt;
    }
    return seeds;
}
}
