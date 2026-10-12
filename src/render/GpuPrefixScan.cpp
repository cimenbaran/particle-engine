#include "GpuPrefixScan.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <stdexcept>
namespace pw {
GpuPrefixScan::GpuPrefixScan(const std::filesystem::path& assets,std::uint32_t count) {
    if(count>1048576) throw std::invalid_argument("Prefix scan supports at most 1048576 elements");
    try {
        scan_=loadCompute(assets,"cs_scan_blocks");add_=loadCompute(assets,"cs_scan_add");
        params_=bgfx::createUniform("u_scan",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(params_)) throw std::runtime_error("Prefix scan uniform unavailable");
        for(;;) {
            const auto groups=std::max(1u,dispatchGroups(count));
            levels_.push_back({count,groups});auto& level=levels_.back();
            const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
            level.output=bgfx::createDynamicIndexBuffer(std::max(1u,count),flags);
            level.totals=bgfx::createDynamicIndexBuffer(groups,flags);
            if(!bgfx::isValid(level.output) || !bgfx::isValid(level.totals)) throw std::runtime_error("Prefix scan buffer unavailable");
            if(groups==1) break;
            count=groups;
        }
    }catch(...) {release();throw;}
}
GpuPrefixScan::~GpuPrefixScan() {release();}
void GpuPrefixScan::release() {
    for(const auto& level:levels_) for(auto handle:{level.output,level.totals})
        if(bgfx::isValid(handle)) bgfx::destroy(handle);
    for(auto handle:{scan_,add_}) if(bgfx::isValid(handle)) bgfx::destroy(handle);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuPrefixScan::submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle input) {
    if(!bgfx::isValid(input)) throw std::invalid_argument("Prefix scan input is invalid");
    for(const auto& level:levels_) if(input.idx==level.output.idx || input.idx==level.totals.idx)
        throw std::invalid_argument("Prefix scan input must not alias its scratch buffers");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    auto source=input;
    for(const auto& level:levels_) {
        const float parameters[]{float(level.count),0,0,0};bgfx::setUniform(params_,parameters);
        bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,level.output,bgfx::Access::Write);
        bgfx::setBuffer(2,level.totals,bgfx::Access::Write);bgfx::dispatch(view,scan_,level.groups);
        source=level.totals;
    }
    for(std::size_t i=levels_.size()-1;i>0;--i) {
        const auto& level=levels_[i-1];
        const float parameters[]{float(level.count),0,0,0};bgfx::setUniform(params_,parameters);
        bgfx::setBuffer(0,level.output,bgfx::Access::ReadWrite);
        bgfx::setBuffer(1,levels_[i].output,bgfx::Access::Read);bgfx::dispatch(view,add_,level.groups);
    }
}
}
