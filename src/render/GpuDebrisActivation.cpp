#include "GpuDebrisActivation.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/TerrainMass.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
namespace pw {
GpuDebrisActivation::GpuDebrisActivation(const std::filesystem::path& assets,std::uint32_t capacity,
    const MaterialLibrary& materials,std::span<const DebrisSeed> seeds)
    :capacity_(capacity),count_(std::uint32_t(seeds.size())),materialHash_(materials.contentHash) {
    if(capacity>1048576 || seeds.size()>4096 || materials.entries.empty() || materials.entries.size()>65536)
        throw std::invalid_argument("Invalid debris activation capacity/library/seed count");
    std::vector<ParticleGpu> records(std::max<std::size_t>(1,seeds.size()));
    for(unsigned i=0;i<seeds.size();++i) {
        const auto& seed=seeds[i];
        if(seed.material>=materials.entries.size())throw std::invalid_argument("Invalid debris material");
        const auto& material=materials.entries[seed.material].gpu;
        if(material.identity[0]!=seed.material || material.identity[1]!=0)throw std::invalid_argument("Debris activation requires a stable solid material ID");
        for(unsigned a=0;a<3;++a)if(!std::isfinite(seed.position[a]) || !std::isfinite(seed.velocity[a]) ||
            seed.position[a]<(a==1?.0625f:-15.9375f) || seed.position[a]>15.9375f)
            throw std::invalid_argument("Debris seed outside the bounded granular profile");
        if(std::hypot(double(seed.velocity[0]),double(seed.velocity[1]),double(seed.velocity[2]))>12 ||
            !std::isfinite(seed.temperature) || seed.temperature<0 || seed.temperature>1000000 ||
            !std::isfinite(seed.fuel) || seed.fuel<0 || seed.fuel>1)
            throw std::invalid_argument("Invalid debris velocity or thermal state");
        auto& p=records[i];for(unsigned a=0;a<3;++a){p.positionRadius[a]=seed.position[a];p.velocityInvMass[a]=seed.velocity[a];}
        p.positionRadius[3]=terrain::conversionGrainRadius;
        p.velocityInvMass[3]=terrain::conversionMassProfile(material.mechanical[0]).inverseMass;
        p.thermalState[0]=seed.temperature;p.thermalState[1]=seed.fuel;
        p.identity[0]=seed.material;p.identity[1]=1;
    }
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        seeds_=bgfx::createDynamicVertexBuffer(bgfx::copy(records.data(),std::uint32_t(records.size()*sizeof(ParticleGpu))),lanes,BGFX_BUFFER_COMPUTE_READ);
        output_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        owners_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_debris_activate_clear");ownersClear_=loadCompute(assets,"cs_debris_owners");claims_=loadCompute(assets,"cs_debris_claims");
        finalize_=loadCompute(assets,"cs_debris_activate_finalize");apply_=loadCompute(assets,"cs_debris_activate");params_=bgfx::createUniform("u_debris_activation",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(seeds_) || !bgfx::isValid(output_) || !bgfx::isValid(owners_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))throw std::runtime_error("Debris activation allocation failed");
    }catch(...){release();throw;}
}
GpuDebrisActivation::~GpuDebrisActivation(){release();}
void GpuDebrisActivation::release() {
    for(auto h:{seeds_,output_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{owners_,status_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{clear_,ownersClear_,claims_,finalize_,apply_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
void GpuDebrisActivation::submit(std::uint16_t view,const GpuDebrisReservation& reservation) {
    if(submitted_)throw std::logic_error("Debris activation is single-use");
    if(!reservation.submitted())throw std::invalid_argument("Debris reservation has not been submitted");
    if(reservation.capacity()!=capacity_)throw std::invalid_argument("Debris reservation capacity mismatch");
    sourceClaims_=reservation.claims();submitted_=true;bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),float(count_),0,0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,reservation.status(),bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(capacity_) {
        bgfx::setUniform(params_,p);bgfx::setBuffer(0,reservation.snapshot(),bgfx::Access::Read);bgfx::setBuffer(1,owners_,bgfx::Access::Write);bgfx::setBuffer(2,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,ownersClear_,dispatchGroups(capacity_));
    }
    if(count_) {
        bgfx::setUniform(params_,p);bgfx::setBuffer(0,reservation.snapshot(),bgfx::Access::Read);bgfx::setBuffer(1,reservation.claims(),bgfx::Access::Read);
        bgfx::setBuffer(2,owners_,bgfx::Access::ReadWrite);bgfx::setBuffer(3,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,claims_,dispatchGroups(count_));
    }
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,finalize_,1);
    if(capacity_) {
        bgfx::setUniform(params_,p);bgfx::setBuffer(0,reservation.snapshot(),bgfx::Access::Read);bgfx::setBuffer(1,seeds_,bgfx::Access::Read);
        bgfx::setBuffer(2,owners_,bgfx::Access::Read);bgfx::setBuffer(3,status_,bgfx::Access::Read);bgfx::setBuffer(4,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
    }
}
}
