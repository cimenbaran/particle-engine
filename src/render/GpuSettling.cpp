#include "GpuSettling.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
namespace pw {
namespace {
std::uint32_t validate(std::uint32_t n,const SettlingProfile& p) {
    validateSettlingProfile(p);
    if(n>1048576 || p.enterSpeed<0.001f || p.exitSpeed>12.0f)
        throw std::invalid_argument("GPU settling requires capacity <=1048576 and speeds 0.001..12 m/s");
    return n;
}
}
GpuSettling::GpuSettling(const std::filesystem::path& assets,std::uint32_t capacity,SettlingProfile profile)
    :capacity_(validate(capacity,profile)),profile_(profile),keys_(assets,capacity_),sort_(assets,capacity_),ranges_(assets,capacity_),hash_(assets,capacity_) {
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        history_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_),layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        support_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_),layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        reset_=loadCompute(assets,"cs_settling_reset");clear_=loadCompute(assets,"cs_settling_clear");
        supportProgram_=loadCompute(assets,"cs_settling_support");reduce_=loadCompute(assets,"cs_settling_reduce");advance_=loadCompute(assets,"cs_settling_advance");
        params_=bgfx::createUniform("u_settling",bgfx::UniformType::Vec4);ticks_=bgfx::createUniform("u_settling_ticks",bgfx::UniformType::Vec4);
        neighbors_=bgfx::createUniform("u_neighbors",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(history_) || !bgfx::isValid(support_) || !bgfx::isValid(status_) ||
           !bgfx::isValid(params_) || !bgfx::isValid(ticks_) || !bgfx::isValid(neighbors_))throw std::runtime_error("Settling resources unavailable");
    }catch(...){release();throw;}
}
GpuSettling::~GpuSettling(){release();}
void GpuSettling::release() {
    for(auto h:{history_,support_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(status_))bgfx::destroy(status_);
    for(auto h:{reset_,clear_,supportProgram_,reduce_,advance_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,ticks_,neighbors_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuSettling::reset(std::uint16_t view) {
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),profile_.enterSpeed,profile_.exitSpeed,0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,history_,bgfx::Access::Write);bgfx::setBuffer(1,status_,bgfx::Access::Write);
    bgfx::dispatch(view,reset_,dispatchGroups(std::max(1u,capacity_)));initialized_=true;
}
void GpuSettling::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,bgfx::DynamicVertexBufferHandle materials,
                         std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream) {
    if(!bgfx::isValid(state) || !bgfx::isValid(materials) || !bgfx::isValid(upstream) || state.idx==materials.idx ||
       upstream.idx==status_.idx || !materialCount || materialCount>65536)
        throw std::invalid_argument("Invalid settling inputs");
    for(auto h:{history_,support_})if(h.idx==state.idx || h.idx==materials.idx)throw std::invalid_argument("Settling scratch alias");
    if(!initialized_)reset(view);
    keys_.submit(view,state,materials,materialCount);sort_.submit(view,keys_.records());
    ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    bgfx::setBuffer(0,upstream,bgfx::Access::Read);bgfx::setBuffer(1,hash_.metadata(),bgfx::Access::Read);
    bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_)return;
    const float n[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),0.127f};
    const float p[]{float(capacity_),profile_.enterSpeed,profile_.exitSpeed,0};
    const std::array<std::uint32_t,4> ticks{profile_.acquireTicks,profile_.releaseTicks,0,0};
    bgfx::setUniform(neighbors_,n);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);
    bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);
    bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,support_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,supportProgram_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,support_,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,reduce_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setUniform(ticks_,ticks.data());
    bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,support_,bgfx::Access::Read);
    bgfx::setBuffer(2,history_,bgfx::Access::ReadWrite);bgfx::setBuffer(3,status_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,advance_,dispatchGroups(capacity_));
}
}
