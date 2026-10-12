#include "GpuVelocityReconstruction.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace pw {
GpuVelocityReconstruction::GpuVelocityReconstruction(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576) throw std::invalid_argument("Velocity reconstruction capacity exceeds 1048576");
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:outputs_) h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        candidate_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_velocity_clear");reconstruct_=loadCompute(assets,"cs_velocity_reconstruct");apply_=loadCompute(assets,"cs_velocity_apply");
        params_=bgfx::createUniform("u_velocity",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(outputs_[0]) || !bgfx::isValid(outputs_[1]) || !bgfx::isValid(candidate_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))
            throw std::runtime_error("Velocity reconstruction resources unavailable");
    }catch(...) {release();throw;}
}
GpuVelocityReconstruction::~GpuVelocityReconstruction(){release();}
void GpuVelocityReconstruction::release() {
    for(auto h:outputs_) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(candidate_)) bgfx::destroy(candidate_);
    if(bgfx::isValid(status_)) bgfx::destroy(status_);
    for(auto h:{clear_,reconstruct_,apply_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuVelocityReconstruction::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle previous,bgfx::DynamicVertexBufferHandle corrected,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle contactStatus,float dt) {
    if(!bgfx::isValid(previous) || !bgfx::isValid(corrected) || !bgfx::isValid(materials) || !bgfx::isValid(contactStatus) ||
       !materialCount || materialCount>65536 || !std::isfinite(dt) || dt<std::numeric_limits<float>::min() || dt>1 || contactStatus.idx==status_.idx)
        throw std::invalid_argument("Invalid velocity reconstruction inputs or timestep");
    for(auto h:{previous,corrected,materials}) if(h.idx==candidate_.idx) throw std::invalid_argument("Velocity input aliases scratch");
    if(materials.idx==previous.idx || materials.idx==corrected.idx || materials.idx==outputs_[0].idx || materials.idx==outputs_[1].idx)
        throw std::invalid_argument("Velocity materials alias particle state");
    auto output=outputs_[0];if(output.idx==previous.idx || output.idx==corrected.idx) output=outputs_[1];
    if(output.idx==previous.idx || output.idx==corrected.idx) throw std::invalid_argument("Both velocity outputs are bound as inputs");
    output_=output;bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,contactStatus,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_) return;
    const float p[]{float(capacity_),dt,float(materialCount),0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,previous,bgfx::Access::Read);bgfx::setBuffer(1,corrected,bgfx::Access::Read);
    bgfx::setBuffer(2,materials,bgfx::Access::Read);bgfx::setBuffer(3,contactStatus,bgfx::Access::Read);
    bgfx::setBuffer(4,candidate_,bgfx::Access::Write);bgfx::setBuffer(5,status_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,reconstruct_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,corrected,bgfx::Access::Read);bgfx::setBuffer(1,candidate_,bgfx::Access::Read);
    bgfx::setBuffer(2,status_,bgfx::Access::Read);bgfx::setBuffer(3,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
