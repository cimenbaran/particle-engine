#include "GpuDebrisReservation.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <bit>
namespace pw {
GpuDebrisReservation::GpuDebrisReservation(const std::filesystem::path& assets,std::uint32_t capacity)
    :capacity_(capacity),scan_(assets,capacity) {
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        snapshot_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        claims_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity),lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        flags_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_debris_clear");flag_=loadCompute(assets,"cs_debris_flags");finalize_=loadCompute(assets,"cs_debris_finalize");reserve_=loadCompute(assets,"cs_debris_reserve");
        params_=bgfx::createUniform("u_debris",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(snapshot_) || !bgfx::isValid(claims_) || !bgfx::isValid(flags_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))throw std::runtime_error("Debris reservation allocation failed");
    }catch(...){release();throw;}
}
GpuDebrisReservation::~GpuDebrisReservation(){release();}
void GpuDebrisReservation::release() {
    for(auto h:{snapshot_,claims_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{flags_,status_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{clear_,flag_,finalize_,reserve_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
void GpuDebrisReservation::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,std::uint32_t requested) {
    if(!bgfx::isValid(input) || input.idx==snapshot_.idx || input.idx==claims_.idx)throw std::invalid_argument("Invalid debris reservation source");
    const float params[]{float(capacity_),std::bit_cast<float>(requested),0,0};
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setUniform(params_,params);
    bgfx::setBuffer(0,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(capacity_) {
        bgfx::setUniform(params_,params);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,flags_,bgfx::Access::Write);
        bgfx::setBuffer(2,status_,bgfx::Access::ReadWrite);bgfx::setBuffer(3,claims_,bgfx::Access::Write);bgfx::dispatch(view,flag_,dispatchGroups(capacity_));
    }
    scan_.submit(view,flags_);
    bgfx::setUniform(params_,params);bgfx::setBuffer(0,scan_.total(),bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,finalize_,1);
    if(capacity_) {
        bgfx::setUniform(params_,params);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,flags_,bgfx::Access::Read);
        bgfx::setBuffer(2,scan_.output(),bgfx::Access::Read);bgfx::setBuffer(3,status_,bgfx::Access::Read);
        bgfx::setBuffer(4,snapshot_,bgfx::Access::Write);bgfx::setBuffer(5,claims_,bgfx::Access::ReadWrite);bgfx::dispatch(view,reserve_,dispatchGroups(capacity_));
    }
}
}
