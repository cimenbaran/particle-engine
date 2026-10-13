#include "GpuCellHash.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <bit>
#include <stdexcept>
namespace pw {
GpuCellHash::GpuCellHash(const std::filesystem::path& assets,std::uint32_t capacity,std::uint32_t slots,std::uint32_t limit,GranularGridProfile profile)
    :capacity_(capacity),slots_(slots),limit_(limit) {
    if(capacity>1048576 || limit==0 || limit>256) throw std::invalid_argument("Invalid GPU hash capacity/probe limit");
    if(!slots_) slots_=std::bit_ceil(capacity*2+2);
    if(slots_<2 || slots_>4194304 || !std::has_single_bit(slots_)) throw std::invalid_argument("Invalid GPU hash table size");
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        table_=bgfx::createDynamicIndexBuffer(slots_*2,flags);
        meta_=bgfx::createDynamicIndexBuffer(4,flags);metrics_=bgfx::createDynamicIndexBuffer(4,flags);
        clear_=loadCompute(assets,"cs_hash_clear");insert_=loadCompute(assets,gridShader("cs_hash_insert",profile));
        finalize_=loadCompute(assets,"cs_hash_finalize");lookup_=loadCompute(assets,gridShader("cs_hash_lookup",profile));
        params_=bgfx::createUniform("u_hash",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(table_) || !bgfx::isValid(meta_) || !bgfx::isValid(metrics_) || !bgfx::isValid(params_))
            throw std::runtime_error("GPU hash resources unavailable");
    }catch(...) {release();throw;}
}
GpuCellHash::~GpuCellHash(){release();}
void GpuCellHash::release() {
    for(auto h:{table_,meta_,metrics_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{clear_,insert_,finalize_,lookup_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuCellHash::submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle ranges,bgfx::DynamicIndexBufferHandle source) {
    if(!bgfx::isValid(ranges) || !bgfx::isValid(source) || ranges.idx==source.idx)
        throw std::invalid_argument("Invalid GPU hash inputs");
    for(auto h:{table_,meta_,metrics_}) if(h.idx==ranges.idx || h.idx==source.idx)
        throw std::invalid_argument("GPU hash input aliases scratch storage");
    source_=ranges;buildView_=view;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),float(slots_),float(limit_),0};
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,source,bgfx::Access::Read);
    bgfx::setBuffer(1,table_,bgfx::Access::Write);bgfx::setBuffer(2,meta_,bgfx::Access::Write);
    bgfx::setBuffer(3,metrics_,bgfx::Access::Write);bgfx::dispatch(view,clear_,dispatchGroups(slots_));
    if(capacity_) {
        bgfx::setUniform(params_,p);bgfx::setBuffer(0,ranges,bgfx::Access::Read);bgfx::setBuffer(1,source,bgfx::Access::Read);
        bgfx::setBuffer(2,table_,bgfx::Access::ReadWrite);bgfx::setBuffer(3,meta_,bgfx::Access::ReadWrite);
        bgfx::setBuffer(4,metrics_,bgfx::Access::ReadWrite);bgfx::dispatch(view,insert_,dispatchGroups(capacity_));
    }
    bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,meta_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,finalize_,1);
}
void GpuCellHash::lookup(std::uint16_t view,bgfx::DynamicIndexBufferHandle keys,bgfx::DynamicIndexBufferHandle results,std::uint32_t count) {
    if(!bgfx::isValid(source_) || view<buildView_ || count>1048576 || !bgfx::isValid(keys) || !bgfx::isValid(results) || keys.idx==results.idx)
        throw std::invalid_argument("Invalid GPU hash query/order");
    for(auto h:{source_,table_,meta_,metrics_}) if(h.idx==keys.idx || h.idx==results.idx)
        throw std::invalid_argument("GPU hash query aliases index storage");
    if(!count) return;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),float(slots_),float(limit_),float(count)};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,source_,bgfx::Access::Read);bgfx::setBuffer(1,table_,bgfx::Access::Read);
    bgfx::setBuffer(2,meta_,bgfx::Access::Read);bgfx::setBuffer(3,keys,bgfx::Access::Read);
    bgfx::setBuffer(4,results,bgfx::Access::Write);bgfx::dispatch(view,lookup_,dispatchGroups(count));
}
}
