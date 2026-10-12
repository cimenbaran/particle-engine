#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
#include <cstdint>
namespace pw {
// Exact-key lookup for the fixed granular grid. Source ranges/metadata must
// remain alive and immutable until all submitted queries have executed.
class GpuCellHash {
public:
    GpuCellHash(const std::filesystem::path& assets,std::uint32_t capacity,
                std::uint32_t tableSlots=0,std::uint32_t probeLimit=64);
    ~GpuCellHash();
    GpuCellHash(const GpuCellHash&)=delete;
    GpuCellHash& operator=(const GpuCellHash&)=delete;
    // Caller provides capacity*3 range words (minimum 3) and 4 metadata words
    // in compute-readable buffers. Metadata follows GpuCellRanges' contract.
    // Use normal ascending view order; finish queries before rebuilding again.
    void submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle ranges,
                bgfx::DynamicIndexBufferHandle sourceMetadata);
    // Each result has four words: begin,end,status,probes. Status is
    // 0 absent, 1 found, 2 failed index, 3 corrupt payload. Failed != absent.
    // Keys/results need count/count*4 words. No allocation or CPU readback.
    // Schedule at or after the build view; buffers must outlive GPU execution.
    void lookup(std::uint16_t view,bgfx::DynamicIndexBufferHandle keys,
                bgfx::DynamicIndexBufferHandle results,std::uint32_t count);
    // ready, inserted, failed insertions, error bits. Only ready==1 permits use.
    bgfx::DynamicIndexBufferHandle metadata() const {return meta_;}
    // total probes, maximum probes, reserved zero, reserved zero.
    bgfx::DynamicIndexBufferHandle metrics() const {return metrics_;}
private:
    std::uint32_t capacity_,slots_,limit_;
    std::uint16_t buildView_=0;
    bgfx::DynamicIndexBufferHandle source_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle table_=BGFX_INVALID_HANDLE,meta_=BGFX_INVALID_HANDLE,metrics_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,insert_=BGFX_INVALID_HANDLE,finalize_=BGFX_INVALID_HANDLE,lookup_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuCellHashTest(const std::filesystem::path& assets);
}
