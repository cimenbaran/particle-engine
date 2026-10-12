#include "GpuGranularStep.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace pw {
namespace {
std::uint32_t validate(std::uint32_t capacity,const GranularStepProfile& p,std::uint32_t iterations,float gravity) {
    const auto plan=planGranularStep(p);
    if(capacity>1048576 || iterations==0 || iterations>8 || !std::isfinite(gravity) || gravity<0 || gravity>100 ||
       plan.substeps>8 || p.smallestFeature>0.125f || p.speedLimit<0.001f || p.speedLimit>1000000 || plan.substepSeconds<std::numeric_limits<float>::min())
        throw std::invalid_argument("Invalid granular solver profile");
    return capacity;
}
}
GpuGranularStep::GpuGranularStep(const std::filesystem::path& assets,std::uint32_t capacity,GranularStepProfile profile,std::uint32_t iterations,float gravity,bool latchFailures)
    :capacity_(validate(capacity,profile,iterations,gravity)),iterations_(iterations),profile_(profile),plan_(planGranularStep(profile)),gravity_(gravity),latchFailures_(latchFailures),
     sweep_(assets,capacity_),contacts_(assets,capacity_,0.8f,0.03125f,true),velocity_(assets,capacity_),response_(assets,capacity_,iterations*0.03125f) {
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:outputs_)h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        forced_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        finished_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        const std::array<std::uint32_t,4> zero{};
        zero_=bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ);
        latched_=bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        neighborhood_=bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        phaseStatus_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_velocity_clear");apply_=loadCompute(assets,"cs_velocity_apply");
        force_=loadCompute(assets,"cs_granular_force");finish_=loadCompute(assets,"cs_granular_finish");collect_=loadCompute(assets,"cs_granular_collect");latch_=loadCompute(assets,"cs_granular_latch");
        params_=bgfx::createUniform("u_velocity",bgfx::UniformType::Vec4);step_=bgfx::createUniform("u_granular",bgfx::UniformType::Vec4);
        collection_=bgfx::createUniform("u_collect",bgfx::UniformType::Vec4);
        for(auto h:outputs_)if(!bgfx::isValid(h))throw std::runtime_error("Granular output allocation failed");
        if(!bgfx::isValid(neighborhood_) || !bgfx::isValid(forced_) || !bgfx::isValid(finished_) || !bgfx::isValid(zero_) || !bgfx::isValid(status_) || !bgfx::isValid(phaseStatus_) || !bgfx::isValid(latched_) ||
           !bgfx::isValid(params_) || !bgfx::isValid(step_) || !bgfx::isValid(collection_))throw std::runtime_error("Granular resources unavailable");
    }catch(...){release();throw;}
}
GpuGranularStep::~GpuGranularStep(){release();}
void GpuGranularStep::release() {
    for(auto h:outputs_)if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{forced_,finished_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{zero_,status_,phaseStatus_,latched_,neighborhood_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{clear_,force_,finish_,collect_,apply_,latch_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,step_,collection_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuGranularStep::collect(std::uint16_t view,bgfx::DynamicIndexBufferHandle source,float phase,float mode) {
    const float p[]{phase,mode,0,0};bgfx::setUniform(collection_,p);
    bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);bgfx::setBuffer(2,neighborhood_,bgfx::Access::ReadWrite);bgfx::dispatch(view,collect_,1);
}
void GpuGranularStep::resetFailures(std::uint16_t view) {
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,latched_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,neighborhood_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
}
void GpuGranularStep::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,std::uint32_t count) {
    if(!bgfx::isValid(input) || !bgfx::isValid(materials) || input.idx==materials.idx || !count || count>65536)
        throw std::invalid_argument("Invalid granular step input");
    for(auto h:{forced_,finished_})if(h.idx==input.idx || h.idx==materials.idx)throw std::invalid_argument("Granular scratch alias");
    for(auto h:outputs_)if(h.idx==materials.idx)throw std::invalid_argument("Granular material/output alias");
    output_=input.idx==outputs_[0].idx?outputs_[1]:outputs_[0];bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,latchFailures_?latched_:zero_,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,neighborhood_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_)return;
    const float p[]{float(capacity_),plan_.substepSeconds,float(count),0},q[]{gravity_,profile_.speedLimit,0,0};
    auto current=input;
    for(std::uint32_t substep=0;substep<plan_.substeps;++substep) {
        bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,phaseStatus_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
        bgfx::setUniform(params_,p);bgfx::setUniform(step_,q);
        bgfx::setBuffer(0,current,bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Read);
        bgfx::setBuffer(3,forced_,bgfx::Access::Write);bgfx::setBuffer(4,phaseStatus_,bgfx::Access::ReadWrite);bgfx::dispatch(view,force_,dispatchGroups(capacity_));collect(view,phaseStatus_,1,0);
        sweep_.submit(view,forced_,materials,count,status_,plan_.substepSeconds,profile_.speedLimit);collect(view,sweep_.status(),2,1);
        const auto substepStart=current;
        const auto predicted=sweep_.output();current=predicted;
        for(std::uint32_t i=0;i<iterations_;++i){contacts_.submit(view,current,materials,count,substepStart);current=contacts_.output();collect(view,contacts_.status(),4,2);}
        velocity_.submit(view,predicted,current,materials,count,status_,plan_.substepSeconds,VelocityReconstructionMode::Correction);collect(view,velocity_.status(),8,0);
        response_.submit(view,predicted,velocity_.output(),materials,count,status_);collect(view,response_.status(),16,2);
        bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,phaseStatus_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
        bgfx::setUniform(params_,p);bgfx::setUniform(step_,q);
        bgfx::setBuffer(0,response_.output(),bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Read);
        bgfx::setBuffer(3,finished_,bgfx::Access::Write);bgfx::setBuffer(4,phaseStatus_,bgfx::Access::ReadWrite);bgfx::dispatch(view,finish_,dispatchGroups(capacity_));collect(view,phaseStatus_,32,1);
        current=finished_;
    }
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,current,bgfx::Access::Read);
    bgfx::setBuffer(2,status_,bgfx::Access::Read);bgfx::setBuffer(3,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
    if(latchFailures_){bgfx::setBuffer(0,status_,bgfx::Access::Read);bgfx::setBuffer(1,latched_,bgfx::Access::ReadWrite);bgfx::dispatch(view,latch_,1);}
}
}
