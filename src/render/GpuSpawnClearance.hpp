#pragma once
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include "core/Commands.hpp"
#include "core/SpawnPlacement.hpp"
namespace pw {
// Read-only overlap classification for lattice candidates against a frozen
// common-radius solid pool. Rebuilds a Basin index on every submission.
class GpuSpawnClearance {
public:
    GpuSpawnClearance(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuSpawnClearance();
    GpuSpawnClearance(const GpuSpawnClearance&)=delete;
    GpuSpawnClearance& operator=(const GpuSpawnClearance&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const Command& spawn);
    // One uvec4/request: code, visited particle count, hash probes, reserved.
    // 0 clear, 1 overlap, 2 invalid index/profile, 3 outside grid, 4 no lattice site.
    // Only the submitted request count is initialized. Input must remain frozen
    // until execution completes; output is valid until the next submission.
    std::uint32_t capacity() const{return capacity_;}
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
private:
    void release();
    std::uint32_t capacity_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    bgfx::DynamicVertexBufferHandle output_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle validate_=BGFX_INVALID_HANDLE,query_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle config_=BGFX_INVALID_HANDLE,command_=BGFX_INVALID_HANDLE,first_=BGFX_INVALID_HANDLE,size_=BGFX_INVALID_HANDLE;
};
void runGpuSpawnClearanceTest(const std::filesystem::path& assets);
}
