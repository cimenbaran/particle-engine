#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
// Initial granular profile: origin zero, width 1/8 m, [-8,8)x[0,16)x[-8,8).
// Bind only the physical pool. Alive solid materials participate; gas/liquid
// and dead slots emit the invalid-key sentinel while retaining the slot ID.
class GpuSpatialKeys {
public:
    GpuSpatialKeys(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuSpatialKeys();
    GpuSpatialKeys(const GpuSpatialKeys&)=delete;
    GpuSpatialKeys& operator=(const GpuSpatialKeys&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount);
    bgfx::DynamicIndexBufferHandle records() const {return records_;}
    // emitted, excluded, invalid position/material/identity, outside domain.
    // Consumers MUST reject an index if either of the last two counters is nonzero.
    bgfx::DynamicIndexBufferHandle status() const {return status_;}
private:
    std::uint32_t capacity_;
    bgfx::DynamicIndexBufferHandle records_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,emit_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuSpatialKeysTest(const std::filesystem::path& assets);
}
