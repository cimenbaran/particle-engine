#pragma once
#include "GpuDebrisActivation.hpp"
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include "GpuVoxelWorld.hpp"
namespace pw {
// Read-only geometric gate for activated private debris. Only new slots query
// overlaps; existing grain contacts may legitimately contain solver overlap.
class GpuDebrisClearance {
public:
    GpuDebrisClearance(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuDebrisClearance();
    GpuDebrisClearance(const GpuDebrisClearance&)=delete;
    GpuDebrisClearance& operator=(const GpuDebrisClearance&)=delete;
    // Execute after activation and the candidate occupancy upload. Keep all
    // inputs unchanged/alive until completion, including borrowed claims.
    // Caller owns material-epoch and base-tick validation. No live state is
    // adopted here; outputs remain valid until the next submission.
    void submit(std::uint16_t view,const GpuDebrisActivation& activation,
                bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,
                const VoxelContactContext& terrain);
    // errors (1 activation,2 claim,4 index/profile,8 overlap,16 terrain),
    // requested count, clear count, rejected count. Entire batch needs errors=0.
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
    // Three vec4 lanes per request: pose, claim, uint4(error,visited,probes,terrainKind).
    bgfx::DynamicVertexBufferHandle records() const{return queries_;}
private:
    void release();
    std::uint32_t capacity_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    bgfx::DynamicVertexBufferHandle queries_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle validate_=BGFX_INVALID_HANDLE,prepare_=BGFX_INVALID_HANDLE,pairs_=BGFX_INVALID_HANDLE,terrain_=BGFX_INVALID_HANDLE,clear_=BGFX_INVALID_HANDLE,reduce_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,index_=BGFX_INVALID_HANDLE;
};
void runGpuDebrisClearanceTest(const std::filesystem::path& assets);
}
