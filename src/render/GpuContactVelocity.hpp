#pragma once
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include <array>
namespace pw {
// Material restitution/friction on immutable predicted/reconstructed snapshots.
// The caller retains prediction's incoming velocities through position solving.
class GpuContactVelocity {
public:
    GpuContactVelocity(const std::filesystem::path& assets,std::uint32_t capacity,
        float maxDisplacement=0.25f,float slop=0.0001f,float bounceThreshold=0.5f,GranularGridProfile profile=GranularGridProfile::Legacy);
    ~GpuContactVelocity();
    GpuContactVelocity(const GpuContactVelocity&)=delete;
    GpuContactVelocity& operator=(const GpuContactVelocity&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle predicted,
        bgfx::DynamicVertexBufferHandle reconstructed,bgfx::DynamicVertexBufferHandle materials,
        std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstreamStatus);
    bgfx::DynamicVertexBufferHandle output() const {return output_;}
    // errors (1 upstream, 2 index, 4 snapshot/profile/bound, 8 arithmetic),
    // maximum degree, candidates and probes. Any error copies input unchanged.
    bgfx::DynamicIndexBufferHandle status() const {return status_;}
    // Read-only diagnostic rows, valid until the next submit.
    bgfx::DynamicVertexBufferHandle diagnosticRecords() const{return result_;}
private:
    std::uint32_t capacity_;
    float bound_,slop_,threshold_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle packed_=BGFX_INVALID_HANDLE,result_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,pack_=BGFX_INVALID_HANDLE,degree_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle respond_=BGFX_INVALID_HANDLE,reduce_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,neighbors_=BGFX_INVALID_HANDLE,materials_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuContactVelocityTest(const std::filesystem::path& assets);
}
