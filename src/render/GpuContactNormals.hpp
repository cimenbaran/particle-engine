#pragma once
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include "core/AnalyticSphere.hpp"
namespace pw {
// Diagnostic-only spatial snapshot. Does not advance or modify physics/history.
class GpuContactNormals {
public:
    GpuContactNormals(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuContactNormals();
    GpuContactNormals(const GpuContactNormals&)=delete;
    GpuContactNormals& operator=(const GpuContactNormals&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,AnalyticSphere sphere={});
    // Two vec4 lanes per slot: normal.xyz/signed gap, then bit-preserving uints
    // source (0 none,1 plane,2 sphere,3 particle), feature, error, reserved zero.
    // Errors: 1 failed index, 2 invalid query geometry, 4 failed traversal.
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
private:
    void release();
    std::uint32_t capacity_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    bgfx::DynamicVertexBufferHandle output_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle neighbors_=BGFX_INVALID_HANDLE,sphere_=BGFX_INVALID_HANDLE;
};
void runGpuContactNormalsTest(const std::filesystem::path& assets);
}
