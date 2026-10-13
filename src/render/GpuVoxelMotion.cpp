#include "GpuVoxelMotion.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace pw {
GpuVoxelMotion::GpuVoxelMotion(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576)throw std::invalid_argument("Voxel motion capacity exceeds limit");
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:outputs_)h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        candidate_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_velocity_clear");motion_=loadCompute(assets,"cs_voxel_motion");finish_=loadCompute(assets,"cs_voxel_finish");apply_=loadCompute(assets,"cs_velocity_apply");
        params_=bgfx::createUniform("u_velocity",bgfx::UniformType::Vec4);profile_=bgfx::createUniform("u_voxel_motion",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(outputs_[0]) || !bgfx::isValid(outputs_[1]) || !bgfx::isValid(candidate_) || !bgfx::isValid(status_) || !bgfx::isValid(params_) || !bgfx::isValid(profile_))throw std::runtime_error("Voxel motion resources unavailable");
    }catch(...){release();throw;}
}
GpuVoxelMotion::~GpuVoxelMotion(){release();}
void GpuVoxelMotion::release(){
    for(auto h:outputs_)if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(candidate_))bgfx::destroy(candidate_);if(bgfx::isValid(status_))bgfx::destroy(status_);
    for(auto h:{clear_,motion_,apply_,finish_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,profile_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuVoxelMotion::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
                           std::uint32_t count,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
                           const terrain::SparseVoxelStore& source,terrain::Chunk origin,float dt,float limit,float threshold,std::uint32_t budget) {
    run(view,input,materials,count,upstream,world,source,origin,dt,limit,threshold,budget,false);
}
void GpuVoxelMotion::finish(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
                           std::uint32_t count,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
                           const terrain::SparseVoxelStore& source,terrain::Chunk origin,float limit) {
    run(view,input,materials,count,upstream,world,source,origin,.01f,limit,0,16,true);
}
void GpuVoxelMotion::run(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
                        std::uint32_t count,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
                        const terrain::SparseVoxelStore& source,terrain::Chunk origin,float dt,float limit,float threshold,
                        std::uint32_t budget,bool finish) {
    if(!world.isCurrent(source,origin))throw std::invalid_argument("Stale voxel motion occupancy/origin");
    if(!bgfx::isValid(input) || !bgfx::isValid(materials) || !bgfx::isValid(upstream) || upstream.idx==status_.idx || !count || count>65536 ||
       !std::isfinite(dt) || dt<std::numeric_limits<float>::min() || dt>.01f || !std::isfinite(limit) || limit<.001f || limit>12 ||
       !std::isfinite(threshold) || threshold<0 || !budget || budget>64)
        throw std::invalid_argument("Invalid voxel motion inputs/profile");
    if(input.idx==materials.idx || input.idx==candidate_.idx || input.idx==world.buffer().idx || materials.idx==candidate_.idx || materials.idx==world.buffer().idx)
        throw std::invalid_argument("Voxel motion input alias");
    for(auto h:outputs_)if(materials.idx==h.idx)throw std::invalid_argument("Voxel motion material/output alias");
    // Alternate external-input submissions too, so prediction remains immutable
    // while a following finish pass publishes into the other buffer.
    output_=input.idx==outputs_[0].idx || (input.idx!=outputs_[1].idx && output_.idx==outputs_[0].idx)?outputs_[1]:outputs_[0];
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,upstream,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_)return;
    const float p[]{float(capacity_),dt,float(count),0},q[]{limit,threshold,float(budget),0};bgfx::setUniform(params_,p);bgfx::setUniform(profile_,q);
    bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);bgfx::setBuffer(2,upstream,bgfx::Access::Read);
    bgfx::setBuffer(3,candidate_,bgfx::Access::Write);bgfx::setBuffer(4,status_,bgfx::Access::ReadWrite);bgfx::setBuffer(5,world.buffer(),bgfx::Access::Read);
    bgfx::dispatch(view,finish?finish_:motion_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,input,bgfx::Access::Read);bgfx::setBuffer(1,candidate_,bgfx::Access::Read);
    bgfx::setBuffer(2,status_,bgfx::Access::Read);bgfx::setBuffer(3,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
