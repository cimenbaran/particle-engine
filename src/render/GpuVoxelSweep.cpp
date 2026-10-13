#include "GpuVoxelSweep.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace pw {
GpuVoxelSweep::GpuVoxelSweep(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576)throw std::invalid_argument("Voxel query capacity exceeds limit");
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        output_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        program_=loadCompute(assets,"cs_voxel_sweep");params_=bgfx::createUniform("u_voxel_query",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(output_) || !bgfx::isValid(program_) || !bgfx::isValid(params_))throw std::runtime_error("Voxel query allocation failed");
    }catch(...){release();throw;}
}
GpuVoxelSweep::~GpuVoxelSweep(){release();}
void GpuVoxelSweep::release(){
    if(bgfx::isValid(output_))bgfx::destroy(output_);
    if(bgfx::isValid(program_))bgfx::destroy(program_);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
void GpuVoxelSweep::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle segments,std::array<float,3> lower) {
    if(!bgfx::isValid(segments) || segments.idx==output_.idx)throw std::invalid_argument("Invalid voxel query input buffer");
    for(float value:lower)if(!std::isfinite(value) || value< -64 || value+0.125f>64)throw std::invalid_argument("Invalid voxel lower corner");
    if(!capacity_)return;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{lower[0],lower[1],lower[2],float(capacity_)};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,segments,bgfx::Access::Read);bgfx::setBuffer(1,output_,bgfx::Access::Write);
    bgfx::dispatch(view,program_,dispatchGroups(capacity_));
}
}
