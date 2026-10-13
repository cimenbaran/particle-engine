#pragma once
#include "ParticleTypes.hpp"
#include "Settling.hpp"
#include "SparseVoxelStore.hpp"
#include "TerrainMass.hpp"
#include <algorithm>
#include <span>
#include <vector>
namespace pw::terrain {
struct DepositMaterial {float density=1000;bool solid=true;};
struct DepositClaim {
    SparseVoxelStore::VoxelEdit edit;
    std::array<ParticleId,2> grains;
    ConversionMass mass;
};
struct DepositPlan {
    std::uint64_t revision=0,tick=0;
    std::vector<DepositClaim> claims;
};
// Read-only CPU reference for the future GPU claim pass. Inputs must be one
// immutable, acknowledged particle/settling/terrain epoch. This does not authorize
// retirement: terrain publication and a GPU acknowledgement must precede it.
inline DepositPlan planDeposits(const SparseVoxelStore& source,std::span<const ParticleGpu> particles,
    std::span<const SettlingState> histories,std::span<const DepositMaterial> materials,
    std::uint64_t tick,std::uint32_t quota=32,double tolerance=maximumConversionError) {
    mass_detail::tolerance(tolerance);
    if(particles.size()>4096 || histories.size()!=particles.size() || materials.empty() || materials.size()>256 || quota>512)
        throw std::invalid_argument("Invalid deposit reference bounds");
    std::vector<GrainMass> masses;for(const auto& m:materials)masses.push_back(conversionMassProfile(m.density));
    std::vector<bool> eligible(particles.size()),retired(particles.size());
    for(unsigned i=0;i<particles.size();++i){const auto& p=particles[i];if(p.identity[1]==0)continue;
        if(p.identity[1]!=1 || p.identity[2]!=i || !p.identity[3] || p.identity[0]>=materials.size())throw std::invalid_argument("Invalid deposit particle identity");
        if(!materials[p.identity[0]].solid)continue;
        for(float v:p.positionRadius)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite deposit position");
        for(float v:p.velocityInvMass)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite deposit velocity");
        if(p.positionRadius[3]<=0 || p.positionRadius[3]>1 || p.velocityInvMass[3]<0)throw std::invalid_argument("Invalid deposit mass/radius");
        for(unsigned a=0;a<3;++a)if(std::abs(p.positionRadius[a])>1e6f)throw std::invalid_argument("Deposit particle outside numeric domain");
        const auto& h=histories[i];double speed=0;for(unsigned a=0;a<3;++a)speed+=double(p.velocityInvMass[a])*p.velocityInvMass[a];
        eligible[i]=p.positionRadius[3]==conversionGrainRadius && p.velocityInvMass[3]==masses[p.identity[0]].inverseMass &&
            h.generation==p.identity[3] && h.candidate==1 && h.quietTicks==SettlingProfile{}.acquireTicks && h.unsupportedTicks==0 && speed<=double(SettlingProfile{}.exitSpeed)*SettlingProfile{}.exitSpeed;
    }
    DepositPlan result{source.revision(),tick,{}};
    const auto intersects=[](const ParticleGpu& p,const GlobalVoxel& voxel){double squared=0;for(unsigned a=0;a<3;++a){const double low=double(voxel[a])*voxelMetres,high=low+voxelMetres;const double distance=std::max({low-p.positionRadius[a],double(p.positionRadius[a])-high,0.});squared+=distance*distance;}return squared<double(p.positionRadius[3])*p.positionRadius[3];};
    // Stable slot order resolves competing owners. Target = first grain's cell;
    // the second owner minimizes distance to that cell centre, then stable ID.
    for(unsigned i=0;i<particles.size() && result.claims.size()<quota;++i){if(!eligible[i] || retired[i])continue;
        const auto& first=particles[i];GlobalVoxel target{};bool inside=true;
        for(unsigned a=0;a<3;++a){target[a]=std::int64_t(std::floor(double(first.positionRadius[a])/voxelMetres));inside&=target[a]>=(a==1?0:-128) && target[a]<=127;}
        if(!inside)continue;const auto address=*addressOf(target);const auto occupancy=source.materialAt(address);
        if(!occupancy || *occupancy!=SparseVoxelStore::empty)continue;
        if(std::any_of(result.claims.begin(),result.claims.end(),[&](const auto& c){return c.edit.address==address;}))continue;
        if(target[1]>0){auto below=target;--below[1];const auto support=source.materialAt(*addressOf(below));if(!support || *support==SparseVoxelStore::empty)continue;
            if(*support>=materials.size() || !materials[*support].solid)continue;}
        const auto mass=planGrainDeposit(2,materials[first.identity[0]].density,tolerance);if(!mass.admissible || mass.voxels!=1)continue;
        unsigned second=unsigned(particles.size());double nearest=std::numeric_limits<double>::infinity();
        for(unsigned j=0;j<particles.size();++j){if(j==i || !eligible[j] || retired[j] || particles[j].identity[0]!=first.identity[0])continue;double squared=0;bool local=true;
            for(unsigned a=0;a<3;++a){const double d=double(particles[j].positionRadius[a])-(double(target[a])+.5)*voxelMetres;local&=std::abs(d)<=.1875;squared+=d*d;}
            if(local && squared<nearest){second=j;nearest=squared;}
        }
        if(second==particles.size())continue;
        bool blocked=false;for(unsigned j=0;j<particles.size();++j)if(j!=i && j!=second && !retired[j] && particles[j].identity[1] && materials[particles[j].identity[0]].solid && intersects(particles[j],target)){blocked=true;break;}
        if(blocked)continue;
        result.claims.push_back({{address,first.identity[0]},{{{i,first.identity[3]},{second,particles[second].identity[3]}}},mass});retired[i]=retired[second]=true;
    }
    return result;
}
}
