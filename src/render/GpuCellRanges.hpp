#pragma once
#include "GranularGridProfile.hpp"
#include "GpuPrefixScan.hpp"
namespace pw {
// Sparse (key,begin,end) ranges. Input keys must use the same grid profile.
class GpuCellRanges {
public:
    GpuCellRanges(const std::filesystem::path& assets,std::uint32_t capacity,GranularGridProfile profile=GranularGridProfile::Legacy);
    ~GpuCellRanges();
    GpuCellRanges(const GpuCellRanges&)=delete;
    GpuCellRanges& operator=(const GpuCellRanges&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle sortedRecords,
        bgfx::DynamicIndexBufferHandle sourceStatus);
    bgfx::DynamicIndexBufferHandle ranges() const {return ranges_;}
    // usable ranges, usable records, error bits, raw diagnostic range count.
    // Error bits: input 1, exit 2, order 4, key 8, count 16, slot 32, ownership 64.
    // Any error zeroes usable counts. Consumers must report it, never interpret
    // this failed epoch as successful empty physics space.
    bgfx::DynamicIndexBufferHandle metadata() const {return meta_;}
private:
    std::uint32_t capacity_;
    GpuPrefixScan scan_;
    bgfx::DynamicIndexBufferHandle flags_=BGFX_INVALID_HANDLE,owners_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle ranges_=BGFX_INVALID_HANDLE,meta_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,mark_=BGFX_INVALID_HANDLE,scatter_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle audit_=BGFX_INVALID_HANDLE,finalize_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuCellRangesTest(const std::filesystem::path& assets);
}
