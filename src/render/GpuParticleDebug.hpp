#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
enum class ParticleDebugMode:std::uint32_t {Material,Speed,Cells,Settling,VelocityDirection,ContactNormal};
// Refreshes render instances without advancing or modifying simulation state.
class GpuParticleDebug {
public:
    explicit GpuParticleDebug(const std::filesystem::path& assets);
    ~GpuParticleDebug();
    GpuParticleDebug(const GpuParticleDebug&)=delete;
    GpuParticleDebug& operator=(const GpuParticleDebug&)=delete;
    void submit(std::uint16_t view,std::uint32_t capacity,ParticleDebugMode mode,
        bgfx::DynamicVertexBufferHandle state,bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,
        bgfx::DynamicVertexBufferHandle history,bgfx::DynamicIndexBufferHandle trackingStatus,bgfx::DynamicVertexBufferHandle instances,
        bgfx::DynamicVertexBufferHandle normals=BGFX_INVALID_HANDLE,bgfx::DynamicIndexBufferHandle solverStatus=BGFX_INVALID_HANDLE);
private:
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuParticleDebugTest(const std::filesystem::path& assets);
}
