#include "core/ComputeDispatch.hpp"
#include "ParticleAudit.hpp"
#include "Shaders.hpp"
#include <stdexcept>
namespace pw {
ParticleAudit::ParticleAudit(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(!capacity || capacity>1048576) throw std::runtime_error("Invalid audit capacity");
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        owners_=bgfx::createDynamicIndexBuffer(capacity,flags);result_=bgfx::createDynamicIndexBuffer(4,flags);
        clear_=loadCompute(assets,"cs_audit_clear");membership_=loadCompute(assets,"cs_audit_membership");state_=loadCompute(assets,"cs_audit_state");
        config_=bgfx::createUniform("u_audit",bgfx::UniformType::Vec4);
        sphere_=bgfx::createUniform("u_audit_sphere",bgfx::UniformType::Vec4);
        readback_=std::make_unique<PoolDiagnostics>(assets);
        if(!bgfx::isValid(owners_) || !bgfx::isValid(result_) || !bgfx::isValid(config_) || !bgfx::isValid(sphere_)) throw std::runtime_error("Audit resource allocation failed");
    }catch(...) {release();throw;}
}
void ParticleAudit::release() {
    readback_.reset();
    for(auto h:{owners_,result_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{clear_,membership_,state_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(config_)) bgfx::destroy(config_);
    if(bgfx::isValid(sphere_)) bgfx::destroy(sphere_);
}
ParticleAudit::~ParticleAudit(){release();}
void ParticleAudit::reset(){clearErrors_=true;readback_->beginEpoch();}
void ParticleAudit::prepare(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
                            bgfx::DynamicIndexBufferHandle active,bgfx::DynamicIndexBufferHandle free,
                            bgfx::DynamicIndexBufferHandle pool,std::uint32_t materials,AnalyticSphere sphere) {
    validateAnalyticSphere(sphere);
    const float config[]{float(capacity_),float(materials),clearErrors_?1.0f:0.0f,0};clearErrors_=false;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setUniform(config_,config);bgfx::setBuffer(0,owners_,bgfx::Access::Write);bgfx::setBuffer(1,result_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,clear_,pw::dispatchGroups(capacity_));
    bgfx::setUniform(config_,config);bgfx::setBuffer(0,state,bgfx::Access::Read);
    bgfx::setBuffer(1,active,bgfx::Access::Read);bgfx::setBuffer(2,free,bgfx::Access::Read);bgfx::setBuffer(3,pool,bgfx::Access::Read);
    bgfx::setBuffer(4,owners_,bgfx::Access::ReadWrite);bgfx::setBuffer(5,result_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,membership_,pw::dispatchGroups(capacity_));
    bgfx::setUniform(config_,config);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,owners_,bgfx::Access::Read);
    const float collider[]{sphere.centre[0],sphere.centre[1],sphere.centre[2],sphere.radius};bgfx::setUniform(sphere_,collider);
    bgfx::setBuffer(2,result_,bgfx::Access::ReadWrite);bgfx::dispatch(view,state_,pw::dispatchGroups(capacity_));
}
}
