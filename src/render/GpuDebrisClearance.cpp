#include "GpuDebrisClearance.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
namespace pw {
GpuDebrisClearance::GpuDebrisClearance(const std::filesystem::path& assets,std::uint32_t capacity)
    :capacity_(capacity),keys_(assets,capacity,GranularGridProfile::Basin),sort_(assets,capacity),
     ranges_(assets,capacity,GranularGridProfile::Basin),hash_(assets,capacity,0,64,GranularGridProfile::Basin) {
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        queries_=bgfx::createDynamicVertexBuffer(4096*3,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        validate_=loadCompute(assets,"cs_spawn_validate");prepare_=loadCompute(assets,"cs_debris_clearance_prepare");pairs_=loadCompute(assets,"cs_debris_clearance_pairs_basin");
        terrain_=loadCompute(assets,"cs_debris_clearance_terrain");clear_=loadCompute(assets,"cs_debris_clearance_clear");reduce_=loadCompute(assets,"cs_debris_clearance_reduce");
        params_=bgfx::createUniform("u_debris_clearance",bgfx::UniformType::Vec4);index_=bgfx::createUniform("u_spawn_index",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(queries_) || !bgfx::isValid(status_) || !bgfx::isValid(params_) || !bgfx::isValid(index_))throw std::runtime_error("Debris clearance allocation failed");
    }catch(...){release();throw;}
}
GpuDebrisClearance::~GpuDebrisClearance(){release();}
void GpuDebrisClearance::release() {
    if(bgfx::isValid(queries_))bgfx::destroy(queries_);if(bgfx::isValid(status_))bgfx::destroy(status_);
    for(auto h:{validate_,prepare_,pairs_,terrain_,clear_,reduce_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,index_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuDebrisClearance::submit(std::uint16_t view,const GpuDebrisActivation& activation,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const VoxelContactContext& terrain) {
    if(!activation.submitted() || activation.capacity()!=capacity_ || !bgfx::isValid(materials) || !materialCount || materialCount>65536 ||
        materials.idx==queries_.idx || materials.idx==activation.output().idx || materials.idx==activation.claims().idx || materials.idx==terrain.world.buffer().idx ||
        !terrain.world.isCurrent(terrain.source,terrain.origin))throw std::invalid_argument("Invalid or stale debris clearance input");
    const auto state=activation.output();const auto count=activation.count();
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);keys_.submit(view,state,materials,materialCount);
    const float config[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),0};
    if(capacity_) {
        bgfx::setUniform(index_,config);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,keys_.records(),bgfx::Access::Read);bgfx::setBuffer(2,keys_.status(),bgfx::Access::ReadWrite);bgfx::dispatch(view,validate_,dispatchGroups(capacity_));
    }
    sort_.submit(view,keys_.records());ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    const float params[]{float(capacity_),float(count),float(materialCount),0};
    bgfx::setUniform(params_,params);bgfx::setBuffer(0,activation.status(),bgfx::Access::Read);bgfx::setBuffer(1,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!count)return;
    bgfx::setUniform(params_,params);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,activation.claims(),bgfx::Access::Read);bgfx::setBuffer(2,activation.status(),bgfx::Access::Read);
    bgfx::setBuffer(3,materials,bgfx::Access::Read);bgfx::setBuffer(4,queries_,bgfx::Access::Write);bgfx::dispatch(view,prepare_,dispatchGroups(count));
    bgfx::setUniform(params_,params);bgfx::setUniform(index_,config);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);
    bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,queries_,bgfx::Access::ReadWrite);bgfx::dispatch(view,pairs_,dispatchGroups(count));
    bgfx::setUniform(params_,params);bgfx::setBuffer(0,queries_,bgfx::Access::ReadWrite);bgfx::setBuffer(1,terrain.world.buffer(),bgfx::Access::Read);bgfx::dispatch(view,terrain_,dispatchGroups(count));
    bgfx::setUniform(params_,params);bgfx::setBuffer(0,queries_,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,reduce_,dispatchGroups(count));
}
}
