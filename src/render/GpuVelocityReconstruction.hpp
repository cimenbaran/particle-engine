#pragma once
#include <bgfx/bgfx.h>
#include <array>
#include <filesystem>
#include <cstdint>
namespace pw {
// Reconstructs dynamic-solid velocities from beginning-of-step positions and
// corrected positions. This is not restitution/friction or a full time step.
class GpuVelocityReconstruction {
public:
    GpuVelocityReconstruction(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuVelocityReconstruction();
    GpuVelocityReconstruction(const GpuVelocityReconstruction&)=delete;
    GpuVelocityReconstruction& operator=(const GpuVelocityReconstruction&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle previous,
        bgfx::DynamicVertexBufferHandle corrected,bgfx::DynamicVertexBufferHandle materials,
        std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle contactStatus,float dt);
    bgfx::DynamicVertexBufferHandle output() const {return output_;}
    // errors, proposed updates, excluded/static slots, invalid slots.
    // Any error preserves the entire corrected input, including its velocities.
    bgfx::DynamicIndexBufferHandle status() const {return status_;}
private:
    std::uint32_t capacity_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle candidate_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,reconstruct_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuVelocityReconstructionTest(const std::filesystem::path& assets);
}
