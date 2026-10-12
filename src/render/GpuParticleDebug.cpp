#include "GpuParticleDebug.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <stdexcept>
namespace pw {
GpuParticleDebug::GpuParticleDebug(const std::filesystem::path& assets) {
    program_=loadCompute(assets,"cs_particle_debug");
    params_=bgfx::createUniform("u_particle_debug",bgfx::UniformType::Vec4);
    if(!bgfx::isValid(params_)){bgfx::destroy(program_);throw std::runtime_error("Debug view uniform unavailable");}
}
GpuParticleDebug::~GpuParticleDebug(){if(bgfx::isValid(program_))bgfx::destroy(program_);if(bgfx::isValid(params_))bgfx::destroy(params_);}
void GpuParticleDebug::submit(std::uint16_t view,std::uint32_t capacity,ParticleDebugMode mode,
    bgfx::DynamicVertexBufferHandle state,bgfx::DynamicVertexBufferHandle materials,std::uint32_t count,
    bgfx::DynamicVertexBufferHandle history,bgfx::DynamicIndexBufferHandle status,bgfx::DynamicVertexBufferHandle instances) {
    if(capacity>1048576 || std::uint32_t(mode)>std::uint32_t(ParticleDebugMode::VelocityDirection) || !count || count>65536 ||
       !bgfx::isValid(state) || !bgfx::isValid(materials) || !bgfx::isValid(history) || !bgfx::isValid(status) || !bgfx::isValid(instances) ||
       state.idx==materials.idx || state.idx==history.idx || materials.idx==history.idx ||
       instances.idx==state.idx || instances.idx==materials.idx || instances.idx==history.idx)
        throw std::invalid_argument("Invalid particle debug view inputs");
    if(!capacity)return;
    const float p[]{float(capacity),float(count),float(std::uint32_t(mode)),12.0f};
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,history,bgfx::Access::Read);bgfx::setBuffer(2,status,bgfx::Access::Read);
    bgfx::setBuffer(3,materials,bgfx::Access::Read);bgfx::setBuffer(4,instances,bgfx::Access::Write);
    bgfx::dispatch(view,program_,dispatchGroups(capacity));
}
}
