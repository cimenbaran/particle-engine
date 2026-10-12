#include "core/ComputeDispatch.hpp"
#include "ParticlePool.hpp"
#include "Shaders.hpp"
#include <bit>
#include <stdexcept>
namespace pw {
ParticlePool::ParticlePool(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(!capacity || capacity>1048576) throw std::runtime_error("Pool capacity must be 1-1048576");
    try {
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        counts_=bgfx::createDynamicIndexBuffer(4,flags);
        free_=bgfx::createDynamicIndexBuffer(capacity,flags);
        active_=bgfx::createDynamicIndexBuffer(capacity,flags);
        clear_=loadCompute(assets,"cs_pool_clear"); classify_=loadCompute(assets,"cs_pool_classify");
        spawn_=loadCompute(assets,"cs_pool_spawn"); brush_=loadCompute(assets,"cs_pool_brush");
        pool_=bgfx::createUniform("u_pool",bgfx::UniformType::Vec4);
        min_=bgfx::createUniform("u_brushMin",bgfx::UniformType::Vec4);
        max_=bgfx::createUniform("u_brushMax",bgfx::UniformType::Vec4);
        value_=bgfx::createUniform("u_brushValue",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(counts_) || !bgfx::isValid(free_) || !bgfx::isValid(active_) ||
           !bgfx::isValid(pool_) || !bgfx::isValid(min_) || !bgfx::isValid(max_) || !bgfx::isValid(value_))
            throw std::runtime_error("GPU particle pool allocation failed");
    }catch(...) {release();throw;}
}
void ParticlePool::release() {
    for(auto h:{counts_,free_,active_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{clear_,classify_,spawn_,brush_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{pool_,min_,max_,value_}) if(bgfx::isValid(h)) bgfx::destroy(h);
}
ParticlePool::~ParticlePool() {release();}
void ParticlePool::rebuild(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,bool resetDiagnostics) {
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float params[]{float(capacity_),0,0,resetDiagnostics?1.0f:0.0f};
    bgfx::setUniform(pool_,params);bgfx::setBuffer(0,counts_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,clear_,1);
    bgfx::setUniform(pool_,params);bgfx::setBuffer(0,state,bgfx::Access::Read);
    bgfx::setBuffer(1,counts_,bgfx::Access::ReadWrite);
    bgfx::setBuffer(2,free_,bgfx::Access::Write);bgfx::setBuffer(3,active_,bgfx::Access::Write);
    bgfx::dispatch(view,classify_,pw::dispatchGroups(capacity_));
}
void ParticlePool::apply(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
                        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const Command& command,float radius) {
    if(radius!=0.045f && radius!=0.0625f) throw std::invalid_argument("Unsupported pool radius");
    CommandLimits limits;limits.materialCount=materialCount;
    if(!validate(command,limits,command.tick).empty()) throw std::runtime_error("Invalid pool command");
    if(command.type!=CommandType::SpawnBrush && command.type!=CommandType::ApplyImpulse && command.type!=CommandType::RemoveBrush)
        throw std::runtime_error("Unsupported pool command");
    if(command.type==CommandType::SpawnBrush && command.count==0) return;
    if(command.type==CommandType::SpawnBrush) rebuild(view,state);
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float minimum[]{command.minimum[0],command.minimum[1],command.minimum[2],float(command.material)};
    const float maximum[]{command.maximum[0],command.maximum[1],command.maximum[2],radius};
    const float values[]{command.parameters[0],command.parameters[1],command.parameters[2],command.parameters[3]};
    const float params[]{float(capacity_),command.type==CommandType::SpawnBrush?float(command.count):(command.type==CommandType::RemoveBrush?1.0f:0.0f),float(materialCount),std::bit_cast<float>(std::uint32_t(command.sequence))};
    bgfx::setUniform(pool_,params);bgfx::setUniform(min_,minimum);bgfx::setUniform(max_,maximum);bgfx::setUniform(value_,values);
    bgfx::setBuffer(0,state,bgfx::Access::ReadWrite);
    if(command.type==CommandType::SpawnBrush) {
        bgfx::setBuffer(1,counts_,bgfx::Access::ReadWrite);bgfx::setBuffer(2,free_,bgfx::Access::Read);
        bgfx::setBuffer(3,materials,bgfx::Access::Read);
        bgfx::dispatch(view,spawn_,pw::dispatchGroups(command.count));
    } else bgfx::dispatch(view,brush_,pw::dispatchGroups(capacity_));
}
void ParticlePool::stageShaders(ShaderReload& reload) {
    reload.compute(clear_,"cs_pool_clear");
    reload.compute(classify_,"cs_pool_classify");
    reload.compute(spawn_,"cs_pool_spawn");
    reload.compute(brush_,"cs_pool_brush");
}
}
