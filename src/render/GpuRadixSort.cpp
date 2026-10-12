#include "GpuRadixSort.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <stdexcept>
namespace pw {
GpuRadixSort::GpuRadixSort(const std::filesystem::path& assets,std::uint32_t count):count_(count),scan_(assets,count) {
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        for(auto& record:records_) record=bgfx::createDynamicIndexBuffer(std::max(1u,count)*2,flags);
        flags_=bgfx::createDynamicIndexBuffer(std::max(1u,count),flags);
        flagProgram_=loadCompute(assets,"cs_radix_flags");scatter_=loadCompute(assets,"cs_radix_scatter");
        params_=bgfx::createUniform("u_radix",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(records_[0]) || !bgfx::isValid(records_[1]) || !bgfx::isValid(flags_) || !bgfx::isValid(params_))
            throw std::runtime_error("GPU radix sort resources unavailable");
    }catch(...) {release();throw;}
}
GpuRadixSort::~GpuRadixSort() {release();}
void GpuRadixSort::release() {
    for(auto h:{records_[0],records_[1],flags_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{flagProgram_,scatter_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuRadixSort::submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle records) {
    if(!bgfx::isValid(records)) throw std::invalid_argument("GPU radix input is invalid");
    for(auto h:{records_[0],records_[1],flags_}) if(h.idx==records.idx)
        throw std::invalid_argument("GPU radix input must not alias scratch storage");
    if(!count_) return;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    auto source=records;
    for(unsigned bit=0;bit<32;++bit) {
        const float parameters[]{float(count_),float(bit),0,0};
        bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,source,bgfx::Access::Read);
        bgfx::setBuffer(1,flags_,bgfx::Access::Write);bgfx::dispatch(view,flagProgram_,dispatchGroups(count_));
        scan_.submit(view,flags_);
        bgfx::setUniform(params_,parameters);bgfx::setBuffer(0,source,bgfx::Access::Read);
        bgfx::setBuffer(1,flags_,bgfx::Access::Read);bgfx::setBuffer(2,scan_.output(),bgfx::Access::Read);
        bgfx::setBuffer(3,scan_.total(),bgfx::Access::Read);bgfx::setBuffer(4,records_[bit%2],bgfx::Access::Write);
        bgfx::dispatch(view,scatter_,dispatchGroups(count_));source=records_[bit%2];
    }
}
}
