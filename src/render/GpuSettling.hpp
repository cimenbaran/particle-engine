#pragma once
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include "core/Settling.hpp"
#include "core/AnalyticSphere.hpp"
namespace pw {
// A read-only particle consumer. One submit is one completed fixed tick.
// Grid selects index bounds; support remains floor, particles and optional sphere.
class GpuSettling {
public:
    GpuSettling(const std::filesystem::path& assets,std::uint32_t capacity,SettlingProfile profile={},bool latchFailures=false,AnalyticSphere obstacle={},GranularGridProfile grid=GranularGridProfile::Legacy);
    ~GpuSettling();
    GpuSettling(const GpuSettling&)=delete;
    GpuSettling& operator=(const GpuSettling&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,
        bgfx::DynamicIndexBufferHandle upstreamStatus);
    void reset(std::uint16_t view);
    // One bit-preserving uvec4 per physical slot, matching SettlingState.
    bgfx::DynamicVertexBufferHandle history() const{return history_;}
    // Error bits (1 upstream, 2 index, 4 profile, 8 traversal), candidates,
    // acquiring slots, retained candidates currently missing support.
    bgfx::DynamicIndexBufferHandle status() const{return latchFailures_?latched_:status_;}
private:
    std::uint32_t capacity_;
    SettlingProfile profile_;
    AnalyticSphere obstacle_;
    bool initialized_=false;
    bool latchFailures_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    bgfx::DynamicVertexBufferHandle history_=BGFX_INVALID_HANDLE,support_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE,latched_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle reset_=BGFX_INVALID_HANDLE,clear_=BGFX_INVALID_HANDLE,supportProgram_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle reduce_=BGFX_INVALID_HANDLE,advance_=BGFX_INVALID_HANDLE,latch_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,ticks_=BGFX_INVALID_HANDLE,neighbors_=BGFX_INVALID_HANDLE,sphere_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuSettlingTest(const std::filesystem::path& assets);
}
