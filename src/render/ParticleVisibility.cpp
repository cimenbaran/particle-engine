#include "core/ComputeDispatch.hpp"
#include "ParticleVisibility.hpp"
#include "Shaders.hpp"
#include <stdexcept>
namespace pw {
ParticleVisibility::ParticleVisibility(const std::filesystem::path& assets,std::uint32_t capacity,bool allowIndirect):capacity_(capacity) {
    if(!capacity || capacity>1048576) throw std::runtime_error("Invalid visibility capacity");
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).add(bgfx::Attrib::TexCoord1,4,bgfx::AttribType::Float).end();
        instances_=bgfx::createDynamicVertexBuffer(capacity,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        counts_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_visible_clear");cull_=loadCompute(assets,"cs_visible");
        config_=bgfx::createUniform("u_cull",bgfx::UniformType::Vec4);
        planes_=bgfx::createUniform("u_frustum",bgfx::UniformType::Vec4,6);
        if(allowIndirect && (bgfx::getCaps()->supported&BGFX_CAPS_DRAW_INDIRECT)) {
            args_=loadCompute(assets,"cs_visible_args");indirect_=bgfx::createIndirectBuffer(1);
            if(!bgfx::isValid(indirect_)) throw std::runtime_error("Indirect draw allocation failed");
        }
        if(!bgfx::isValid(instances_) || !bgfx::isValid(counts_) || !bgfx::isValid(config_) || !bgfx::isValid(planes_))
            throw std::runtime_error("Visibility resources unavailable");
    }catch(...) {release();throw;}
}
void ParticleVisibility::release() {
    if(bgfx::isValid(instances_)) bgfx::destroy(instances_);
    if(bgfx::isValid(counts_)) bgfx::destroy(counts_);
    if(bgfx::isValid(indirect_)) bgfx::destroy(indirect_);
    for(auto h:{clear_,cull_,args_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{config_,planes_}) if(bgfx::isValid(h)) bgfx::destroy(h);
}
ParticleVisibility::~ParticleVisibility(){release();}
void ParticleVisibility::prepare(std::uint16_t view,bgfx::DynamicVertexBufferHandle instances,
                                bgfx::DynamicIndexBufferHandle active,bgfx::DynamicIndexBufferHandle pool,
                                const Frustum& planes,bool cull) {
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float config[]{float(capacity_),cull?1.0f:0.0f,0,0};
    bgfx::setUniform(config_,config);bgfx::setBuffer(0,instances_,bgfx::Access::Write);
    bgfx::setBuffer(1,counts_,bgfx::Access::Write);bgfx::dispatch(view,clear_,pw::dispatchGroups(capacity_));
    bgfx::setUniform(config_,config);bgfx::setUniform(planes_,planes.data(),6);
    bgfx::setBuffer(0,instances,bgfx::Access::Read);bgfx::setBuffer(1,instances_,bgfx::Access::Write);
    bgfx::setBuffer(2,counts_,bgfx::Access::ReadWrite);bgfx::setBuffer(3,active,bgfx::Access::Read);
    bgfx::setBuffer(4,pool,bgfx::Access::Read);bgfx::dispatch(view,cull_,pw::dispatchGroups(capacity_));
    if(supportsIndirect()) {
        bgfx::setUniform(config_,config);bgfx::setBuffer(0,counts_,bgfx::Access::Read);
        bgfx::setBuffer(1,indirect_,bgfx::Access::Write);bgfx::dispatch(view,args_,1);
    }
}
void ParticleVisibility::stageShaders(ShaderReload& reload) {
    reload.compute(clear_,"cs_visible_clear");
    reload.compute(cull_,"cs_visible");
    if(supportsIndirect()) reload.compute(args_,"cs_visible_args");
}
}
