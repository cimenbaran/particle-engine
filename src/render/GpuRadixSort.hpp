#pragma once
#include "GpuPrefixScan.hpp"
#include <array>
namespace pw {
// Stable unsigned 32-bit key sort of interleaved (key, slot) uint32 records.
// Persistent particle state is not an input and cannot be reordered here.
class GpuRadixSort {
public:
    GpuRadixSort(const std::filesystem::path& assets,std::uint32_t count);
    ~GpuRadixSort();
    GpuRadixSort(const GpuRadixSort&)=delete;
    GpuRadixSort& operator=(const GpuRadixSort&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle records);
    bgfx::DynamicIndexBufferHandle output() const {return records_[1];}
private:
    std::uint32_t count_;
    GpuPrefixScan scan_;
    std::array<bgfx::DynamicIndexBufferHandle,2> records_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicIndexBufferHandle flags_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle flagProgram_=BGFX_INVALID_HANDLE,scatter_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuRadixSortTest(const std::filesystem::path& assets);
}
