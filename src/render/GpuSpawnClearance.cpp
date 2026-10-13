#include "GpuSpawnClearance.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <bit>
namespace pw {
GpuSpawnClearance::GpuSpawnClearance(const std::filesystem::path& assets,std::uint32_t capacity)
    :capacity_(capacity),keys_(assets,capacity,GranularGridProfile::Basin),sort_(assets,capacity),
    ranges_(assets,capacity,GranularGridProfile::Basin),hash_(assets,capacity,0,64,GranularGridProfile::Basin) {
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        output_=bgfx::createDynamicVertexBuffer(4096,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        validate_=loadCompute(assets,"cs_spawn_validate");query_=loadCompute(assets,"cs_spawn_clearance_basin");
        config_=bgfx::createUniform("u_spawn_index",bgfx::UniformType::Vec4);
        command_=bgfx::createUniform("u_spawn_query",bgfx::UniformType::Vec4);
        first_=bgfx::createUniform("u_spawn_lattice_min",bgfx::UniformType::Vec4);
        size_=bgfx::createUniform("u_spawn_lattice_size",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(output_) || !bgfx::isValid(config_) || !bgfx::isValid(command_) || !bgfx::isValid(first_) || !bgfx::isValid(size_))throw std::runtime_error("Spawn clearance resources unavailable");
    }catch(...){release();throw;}
}
GpuSpawnClearance::~GpuSpawnClearance(){release();}
void GpuSpawnClearance::release() {
    if(bgfx::isValid(output_))bgfx::destroy(output_);
    for(auto h:{validate_,query_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{config_,command_,first_,size_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuSpawnClearance::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const Command& spawn) {
    CommandLimits limits;limits.materialCount=materialCount;
    if(!bgfx::isValid(state) || !bgfx::isValid(materials) || state.idx==materials.idx || state.idx==output_.idx || materials.idx==output_.idx ||
       !materialCount || materialCount>65536 || spawn.type!=CommandType::SpawnBrush || !validate(spawn,limits,spawn.tick).empty())
        throw std::invalid_argument("Invalid spawn clearance input");
    const auto lattice=spawnLattice(spawn.minimum,spawn.maximum);
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    keys_.submit(view,state,materials,materialCount);
    const float config[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),0};
    if(capacity_) {
        bgfx::setUniform(config_,config);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,keys_.records(),bgfx::Access::Read);
        bgfx::setBuffer(2,keys_.status(),bgfx::Access::ReadWrite);bgfx::dispatch(view,validate_,dispatchGroups(capacity_));
    }
    sort_.submit(view,keys_.records());ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    if(!spawn.count)return;
    const float command[]{float(spawn.count),std::bit_cast<float>(std::uint32_t(spawn.sequence)),0,0};
    const float first[]{float(lattice.first[0]),float(lattice.first[1]),float(lattice.first[2]),1};
    const float size[]{float(lattice.dimensions[0]),float(lattice.dimensions[1]),float(lattice.dimensions[2]),std::bit_cast<float>(lattice.sites)};
    bgfx::setUniform(config_,config);bgfx::setUniform(command_,command);bgfx::setUniform(first_,first);bgfx::setUniform(size_,size);
    bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);
    bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,output_,bgfx::Access::Write);
    bgfx::dispatch(view,query_,dispatchGroups(spawn.count));
}
}
