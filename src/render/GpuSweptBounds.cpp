#include "GpuSweptBounds.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace pw {
GpuSweptBounds::GpuSweptBounds(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576) throw std::invalid_argument("Swept bounds capacity exceeds 1048576");
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:outputs_) h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        candidate_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_velocity_clear");sweep_=loadCompute(assets,"cs_swept_bounds");apply_=loadCompute(assets,"cs_velocity_apply");
        params_=bgfx::createUniform("u_velocity",bgfx::UniformType::Vec4);
        sphere_=bgfx::createUniform("u_sweep_sphere",bgfx::UniformType::Vec4);
        profile_=bgfx::createUniform("u_sweep",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(outputs_[0]) || !bgfx::isValid(outputs_[1]) || !bgfx::isValid(candidate_) || !bgfx::isValid(status_) || !bgfx::isValid(params_) || !bgfx::isValid(profile_) || !bgfx::isValid(sphere_))
            throw std::runtime_error("Swept bounds resources unavailable");
    }catch(...) {release();throw;}
}
GpuSweptBounds::~GpuSweptBounds(){release();}
void GpuSweptBounds::release() {
    for(auto h:outputs_) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(candidate_)) bgfx::destroy(candidate_);
    if(bgfx::isValid(status_)) bgfx::destroy(status_);
    for(auto h:{clear_,sweep_,apply_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
    if(bgfx::isValid(profile_)) bgfx::destroy(profile_);
    if(bgfx::isValid(sphere_)) bgfx::destroy(sphere_);
}
void GpuSweptBounds::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream,
    float dt,float speedLimit,float threshold,std::uint32_t budget,const AnalyticSphere& obstacle) {
    validateAnalyticSphere(obstacle);
    if(!bgfx::isValid(input) || !bgfx::isValid(materials) || !bgfx::isValid(upstream) ||
       !materialCount || materialCount>65536 || !std::isfinite(dt) || dt<std::numeric_limits<float>::min() || dt>1 ||
       !std::isfinite(speedLimit) || speedLimit<0.001f || speedLimit>1000000 ||
       !std::isfinite(threshold) || threshold<0 || budget==0 || budget>1024 || upstream.idx==status_.idx)
        throw std::invalid_argument("Invalid swept bounds inputs/profile");
    if(input.idx==candidate_.idx || materials.idx==candidate_.idx || materials.idx==input.idx ||
       materials.idx==outputs_[0].idx || materials.idx==outputs_[1].idx)
        throw std::invalid_argument("Swept bounds input alias");
    output_=input.idx==outputs_[0].idx?outputs_[1]:outputs_[0];
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,upstream,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_) return;
    const float p[]{float(capacity_),dt,float(materialCount),0},q[]{speedLimit,threshold,float(budget),0};
    const float sphere[]{obstacle.centre[0],obstacle.centre[1],obstacle.centre[2],obstacle.radius};
    bgfx::setUniform(sphere_,sphere);bgfx::setUniform(params_,p);bgfx::setUniform(profile_,q);
    bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);
    bgfx::setBuffer(2,upstream,bgfx::Access::Read);bgfx::setBuffer(3,candidate_,bgfx::Access::Write);bgfx::setBuffer(4,status_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,sweep_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,candidate_,bgfx::Access::Read);
    bgfx::setBuffer(2,status_,bgfx::Access::Read);bgfx::setBuffer(3,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
