#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
#include <vector>
namespace pw {
// Exclusive uint32 scan; addition is modulo 2^32. Radix flags are 0/1,
// so their total is bounded by capacity. No synchronous CPU count readback.
class GpuPrefixScan {
public:
    GpuPrefixScan(const std::filesystem::path& assets,std::uint32_t count);
    ~GpuPrefixScan();
    GpuPrefixScan(const GpuPrefixScan&)=delete;
    GpuPrefixScan& operator=(const GpuPrefixScan&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle input);
    bgfx::DynamicIndexBufferHandle output() const {return levels_.front().output;}
    bgfx::DynamicIndexBufferHandle total() const {return levels_.back().totals;}
private:
    struct Level {
        std::uint32_t count,groups;
        bgfx::DynamicIndexBufferHandle output=BGFX_INVALID_HANDLE,totals=BGFX_INVALID_HANDLE;
    };
    std::vector<Level> levels_;
    bgfx::ProgramHandle scan_=BGFX_INVALID_HANDLE,add_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuPrefixScanTest(const std::filesystem::path& assets);
}
