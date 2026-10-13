#pragma once
#include "SparseVoxelStore.hpp"
#include "TerrainMass.hpp"
#include "TerrainDebrisPlacement.hpp"
#include <algorithm>
#include <array>
#include <span>
#include <vector>

namespace pw::terrain {
// A separate typed request avoids overloading the box-brush parameter lanes.
// Energy covers this terrain batch only. Existing-particle impulses require a
// separately allocated share of the command budget, never this same energy.
struct ExplosionRequest {
    std::uint64_t tick=0,sequence=0;
    std::array<float,3> centre{};
    float radius=1,impulse=2; // metres; maximum debris speed in m/s
    double energyJoules=0;
    std::uint32_t voxelQuota=32,grainQuota=1024;
};
struct ExplosionMaterial {
    float density=1000;
    bool solid=true;
    double resistance=1000; // calibrated fracture work, J/m^3
    float fuel=0;
};
struct ExplosionPlan {
    enum class State {Empty,Ready,NoCapacity,InsufficientEnergy,NoSpace};
    State state=State::Empty;
    std::uint64_t revision=0,tick=0,sequence=0;
    std::uint32_t eligible=0,capacityDeferred=0,energyRejected=0,placementDeferred=0,grains=0;
    double reservedJoules=0,fractureJoules=0,kineticJoules=0;
    std::vector<SparseVoxelStore::VoxelEdit> removals;
    std::vector<std::uint32_t> grainsPerMaterial;
    std::vector<DebrisSeed> seeds;
};
inline void validateExplosion(const ExplosionRequest& request,std::uint64_t tick) {
    if(request.tick!=tick || request.sequence==UINT64_MAX ||
       !std::isfinite(request.radius) || request.radius<=0 || request.radius>4 ||
       !std::isfinite(request.impulse) || request.impulse<0 || request.impulse>12 ||
       !std::isfinite(request.energyJoules) || request.energyJoules<0 || request.energyJoules>1e9 ||
       !request.voxelQuota || request.voxelQuota>512 || request.grainQuota>4096)
        throw std::invalid_argument("Invalid explosion tick, radius, impulse, energy or quota");
    for(unsigned a=0;a<3;++a)if(!std::isfinite(request.centre[a]) || request.centre[a]<(a==1?0:-16) || request.centre[a]>16)
        throw std::invalid_argument("Explosion centre outside the resident basin");
}
// Arithmetic helper only: call after successful publication. Round the
// remainder down so repeated batches cannot create energy through subtraction.
inline double explosionEnergyRemainder(double remaining,double committedCharge) {
    if(!std::isfinite(remaining) || remaining<0 || remaining>1e9 ||
       !std::isfinite(committedCharge) || committedCharge<0 || committedCharge>remaining)
        throw std::invalid_argument("Invalid committed explosion energy charge");
    if(committedCharge==0)return remaining;
    return std::nextafter(remaining-committedCharge,0.);
}
// Read-only removal/capacity/energy proposal. Local placement is included. The caller must
// reserve slots and obtain the joint GPU acknowledgement before applying edits
// or charging reservedJoules. Retry a failed batch with the unchanged budget.
inline ExplosionPlan planExplosion(const SparseVoxelStore& source,const ExplosionRequest& request,
    std::uint64_t tick,std::span<const ExplosionMaterial> materials) {
    validateExplosion(request,tick);
    if(materials.empty() || materials.size()>65536)throw std::invalid_argument("Invalid explosion materials");
    for(const auto& material:materials) {
        conversionMassProfile(material.density);
        if(!std::isfinite(material.resistance) || material.resistance<=0 || material.resistance>1e12 ||
           !std::isfinite(material.fuel) || material.fuel<0 || material.fuel>1)
            throw std::invalid_argument("Invalid explosion material resistance");
    }
    ExplosionPlan plan;plan.revision=source.revision();plan.tick=tick;plan.sequence=request.sequence;
    plan.grainsPerMaterial.resize(materials.size());
    struct Candidate {VoxelAddress address;unsigned material;double distanceSquared,falloff;int x,y,z;};
    std::vector<Candidate> candidates;
    std::array<int,3> lo,hi;
    for(unsigned a=0;a<3;++a) {
        lo[a]=std::max(a==1?0:-128,int(std::ceil((double(request.centre[a])-request.radius)*8-.5)));
        hi[a]=std::min(127,int(std::floor((double(request.centre[a])+request.radius)*8-.5)));
    }
    const double radiusSquared=double(request.radius)*request.radius;
    // At most 65^3 centre checks for the validated four-metre radius. Empty and
    // nonresident chunks have no candidates; the store is never expanded here.
    for(int y=lo[1];y<=hi[1];++y)for(int z=lo[2];z<=hi[2];++z)for(int x=lo[0];x<=hi[0];++x) {
        const double dx=(x+.5)*voxelMetres-request.centre[0],dy=(y+.5)*voxelMetres-request.centre[1],dz=(z+.5)*voxelMetres-request.centre[2];
        const double distance=dx*dx+dy*dy+dz*dz;
        if(distance>=radiusSquared)continue; // Zero falloff at the boundary.
        const auto address=*addressOf({x,y,z});const auto material=source.materialAt(address);
        if(!material || *material==SparseVoxelStore::empty)continue;
        if(*material>=materials.size() || !materials[*material].solid)throw std::invalid_argument("Explosion requires solid resident terrain");
        const double linear=1-std::sqrt(distance)/request.radius;
        candidates.push_back({address,*material,distance,linear*linear,x,y,z});
    }
    std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b) {
        if(a.distanceSquared!=b.distanceSquared)return a.distanceSquared<b.distanceSquared;
        if(a.y!=b.y)return a.y>b.y;
        if(a.z!=b.z)return a.z<b.z;
        return a.x<b.x;
    });
    plan.eligible=std::uint32_t(candidates.size());
    std::vector<std::uint32_t> voxels(materials.size());
    const auto upward=[](double value){return std::nextafter(value,std::numeric_limits<double>::infinity());};
    for(const auto& candidate:candidates) {
        const auto m=candidate.material;
        const auto conversion=planVoxelDebris(voxels[m]+1,materials[m].density);
        const auto extra=conversion.grains-plan.grainsPerMaterial[m];
        if(plan.removals.size()==request.voxelQuota || !conversion.admissible || plan.grains+extra>request.grainQuota) {++plan.capacityDeferred;continue;}
        // Quadratic attenuation raises the work required near the edge. Reserve
        // kinetic energy for every quantized product at the maximum speed, so
        // later local placement / radial falloff cannot exceed this allowance.
        const double fracture=upward(upward(materials[m].resistance*(voxelMetres*voxelMetres*voxelMetres))/candidate.falloff);
        const double mass=upward(double(extra)/conversion.inverseGrainMass);
        const double kinetic=request.impulse==0?0:upward(upward(upward(.5*mass)*request.impulse)*request.impulse);
        const double charge=upward(upward(fracture)+upward(kinetic));
        const double total=upward(plan.reservedJoules+charge);
        if(total>request.energyJoules) {++plan.energyRejected;continue;}
        plan.reservedJoules=total;plan.fractureJoules+=fracture;plan.kineticJoules+=kinetic;
        ++voxels[m];plan.grainsPerMaterial[m]=conversion.grains;plan.grains+=extra;
        plan.removals.push_back({candidate.address,SparseVoxelStore::empty});
    }
    if(!plan.removals.empty()) {
        std::vector<DebrisMaterial> profiles;for(const auto& m:materials)profiles.push_back({m.density,m.solid,m.fuel});
        auto placed=placeTerrainDebris(source,plan.removals,profiles);
        if(!placed) {
            plan.state=ExplosionPlan::State::NoSpace;plan.placementDeferred=unsigned(plan.removals.size());
            plan.removals.clear();plan.grains=0;std::fill(plan.grainsPerMaterial.begin(),plan.grainsPerMaterial.end(),0);
            plan.reservedJoules=plan.fractureJoules=plan.kineticJoules=0;
            return plan;
        }
        plan.seeds=std::move(*placed);
        for(auto& seed:plan.seeds) {
            std::array<double,3> direction;for(unsigned a=0;a<3;++a)direction[a]=double(seed.position[a])-request.centre[a];
            const double distance=std::hypot(direction[0],direction[1],direction[2]);
            const double linear=std::max(0.,1-distance/request.radius),speed=request.impulse*linear*linear;
            if(distance==0)seed.velocity={0,float(speed),0};
            else for(unsigned a=0;a<3;++a)seed.velocity[a]=float(direction[a]*speed/distance);
            // Component rounding must not turn an allowed 12 m/s request into
            // an activation rejection or exceed the reserved kinetic ceiling.
            while(std::hypot(double(seed.velocity[0]),double(seed.velocity[1]),double(seed.velocity[2]))>speed)
                for(float& component:seed.velocity)component=std::nextafter(component,0.f);
        }
        plan.state=ExplosionPlan::State::Ready;
    }
    else if(plan.eligible)plan.state=plan.capacityDeferred?ExplosionPlan::State::NoCapacity:ExplosionPlan::State::InsufficientEnergy;
    return plan;
}
}
