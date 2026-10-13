#include "GpuCellRanges.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <stdexcept>
namespace pw {
GpuCellRanges::GpuCellRanges(const std::filesystem::path& assets,std::uint32_t capacity,GranularGridProfile profile):capacity_(capacity),scan_(assets,capacity) {
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        flags_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity),flags);
        owners_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity),flags);
        ranges_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity)*3,flags);
        meta_=bgfx::createDynamicIndexBuffer(4,flags);
        clear_=loadCompute(assets,"cs_range_clear");mark_=loadCompute(assets,gridShader("cs_range_flags",profile));
        scatter_=loadCompute(assets,gridShader("cs_range_scatter",profile));audit_=loadCompute(assets,"cs_range_owners");
        finalize_=loadCompute(assets,"cs_range_finalize");params_=bgfx::createUniform("u_ranges",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(flags_) || !bgfx::isValid(owners_) || !bgfx::isValid(ranges_) || !bgfx::isValid(meta_) || !bgfx::isValid(params_))
            throw std::runtime_error("GPU range resources unavailable");
    }catch(...) {release();throw;}
}
GpuCellRanges::~GpuCellRanges() {release();}
void GpuCellRanges::release() {
    for(auto handle:{flags_,owners_,ranges_,meta_}) if(bgfx::isValid(handle)) bgfx::destroy(handle);
    for(auto handle:{clear_,mark_,scatter_,audit_,finalize_}) if(bgfx::isValid(handle)) bgfx::destroy(handle);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuCellRanges::submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle records,bgfx::DynamicIndexBufferHandle source) {
    if(!bgfx::isValid(records) || !bgfx::isValid(source) || records.idx==source.idx)
        throw std::invalid_argument("Invalid GPU range inputs");
    for(auto h:{flags_,owners_,ranges_,meta_}) if(h.idx==records.idx || h.idx==source.idx)
        throw std::invalid_argument("GPU range input aliases scratch storage");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float parameters[]{float(capacity_),0,0,0};bgfx::setUniform(params_,parameters);
    bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,meta_,bgfx::Access::ReadWrite);
    bgfx::setBuffer(2,owners_,bgfx::Access::Write);bgfx::dispatch(view,clear_,std::max(1u,dispatchGroups(capacity_)));
    if(capacity_) {
        bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,records,bgfx::Access::Read);
        bgfx::setBuffer(1,flags_,bgfx::Access::Write);bgfx::setBuffer(2,meta_,bgfx::Access::ReadWrite);
        bgfx::setBuffer(3,owners_,bgfx::Access::ReadWrite);bgfx::dispatch(view,mark_,dispatchGroups(capacity_));
    }
    scan_.submit(view,flags_);
    if(capacity_) {
        bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,records,bgfx::Access::Read);
        bgfx::setBuffer(1,flags_,bgfx::Access::Read);bgfx::setBuffer(2,scan_.output(),bgfx::Access::Read);
        bgfx::setBuffer(3,ranges_,bgfx::Access::Write);bgfx::dispatch(view,scatter_,dispatchGroups(capacity_));
        bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,owners_,bgfx::Access::Read);
        bgfx::setBuffer(1,meta_,bgfx::Access::ReadWrite);bgfx::dispatch(view,audit_,dispatchGroups(capacity_));
    }
    bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,source,bgfx::Access::Read);
    bgfx::setBuffer(1,meta_,bgfx::Access::ReadWrite);bgfx::setBuffer(2,scan_.total(),bgfx::Access::Read);
    bgfx::dispatch(view,finalize_,1);
}
}
