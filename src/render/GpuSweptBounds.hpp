#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
#include <array>
namespace pw {
class GpuSweptBounds {
public:
    GpuSweptBounds(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuSweptBounds();
    GpuSweptBounds(const GpuSweptBounds&)=delete;
    GpuSweptBounds& operator=(const GpuSweptBounds&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,
        bgfx::DynamicIndexBufferHandle upstream,float dt,float speedLimit=12,
        float bounceThreshold=0.5f,std::uint32_t eventBudget=16);
    bgfx::DynamicVertexBufferHandle output() const {return output_;}
    // errors (1 upstream, 2 profile/state, 4 exhaustion, 8 arithmetic),
    // proposed capped bodies, total collision events, total incident planes.
    bgfx::DynamicIndexBufferHandle status() const {return status_;}
private:
    void release();
    std::uint32_t capacity_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle candidate_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,sweep_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,profile_=BGFX_INVALID_HANDLE;
};
void runGpuSweptBoundsTest(const std::filesystem::path& assets);
}
