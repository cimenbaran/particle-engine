#include "GpuSpatialKeys.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <stdexcept>
namespace pw {
GpuSpatialKeys::GpuSpatialKeys(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576) throw std::invalid_argument("Spatial keys support at most 1048576 slots");
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        records_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity)*2,flags);
        status_=bgfx::createDynamicIndexBuffer(4,flags);
        clear_=loadCompute(assets,"cs_key_clear");emit_=loadCompute(assets,"cs_spatial_keys");
        params_=bgfx::createUniform("u_keys",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(records_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))
            throw std::runtime_error("Spatial key resources unavailable");
    }catch(...) {release();throw;}
}
GpuSpatialKeys::~GpuSpatialKeys() {release();}
void GpuSpatialKeys::release() {
    for(auto handle:{records_,status_}) if(bgfx::isValid(handle)) bgfx::destroy(handle);
    for(auto handle:{clear_,emit_}) if(bgfx::isValid(handle)) bgfx::destroy(handle);
    if(bgfx::isValid(params_)) bgfx::destroy(params_);
}
void GpuSpatialKeys::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount) {
    if(!bgfx::isValid(state) || !bgfx::isValid(materials) || !materialCount || materialCount>65536)
        throw std::invalid_argument("Invalid spatial key input buffers or material count");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,clear_,1);
    if(!capacity_) return;
    const float parameters[]{float(capacity_),float(materialCount),0,0};bgfx::setUniform(params_,parameters);
    bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);
    bgfx::setBuffer(2,records_,bgfx::Access::Write);bgfx::setBuffer(3,status_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,emit_,dispatchGroups(capacity_));
}
}
