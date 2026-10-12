#include "GpuCellOccupancy.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <array>
#include <stdexcept>
namespace pw {
GpuCellOccupancy::GpuCellOccupancy(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity){
    if(capacity>1048576)throw std::invalid_argument("Invalid occupancy capacity");
    try{
        const std::array<std::uint32_t,4> zero{};
        status_=bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_occupancy_clear");reduce_=loadCompute(assets,"cs_occupancy_reduce");
        params_=bgfx::createUniform("u_occupancy",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(status_) || !bgfx::isValid(params_))throw std::runtime_error("Occupancy resources unavailable");
    }catch(...){release();throw;}
}
GpuCellOccupancy::~GpuCellOccupancy(){release();}
void GpuCellOccupancy::release(){
    if(bgfx::isValid(status_))bgfx::destroy(status_);
    for(auto h:{clear_,reduce_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
void GpuCellOccupancy::reset(std::uint16_t view,bgfx::DynamicIndexBufferHandle zero){
    if(!bgfx::isValid(zero) || zero.idx==status_.idx)throw std::invalid_argument("Invalid occupancy reset source");
    const float p[]{float(capacity_),1,0,0};bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,zero,bgfx::Access::Read);bgfx::setBuffer(1,zero,bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
}
void GpuCellOccupancy::submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle ranges,bgfx::DynamicIndexBufferHandle metadata,bgfx::DynamicIndexBufferHandle upstream){
    for(auto h:{ranges,metadata,upstream})if(!bgfx::isValid(h) || h.idx==status_.idx)throw std::invalid_argument("Invalid occupancy input");
    const float p[]{float(capacity_),0,0,0};bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,metadata,bgfx::Access::Read);bgfx::setBuffer(1,upstream,bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_)return;
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,ranges,bgfx::Access::Read);bgfx::setBuffer(1,metadata,bgfx::Access::Read);
    bgfx::setBuffer(2,upstream,bgfx::Access::Read);bgfx::setBuffer(3,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,reduce_,dispatchGroups(capacity_));
}
}
