#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
// Read-only reduction of validated sparse cell ranges. Ignore counts on error.
class GpuCellOccupancy {
public:
    GpuCellOccupancy(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuCellOccupancy();
    GpuCellOccupancy(const GpuCellOccupancy&)=delete;
    GpuCellOccupancy& operator=(const GpuCellOccupancy&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicIndexBufferHandle ranges,
        bgfx::DynamicIndexBufferHandle metadata,bgfx::DynamicIndexBufferHandle upstream);
    void reset(std::uint16_t view,bgfx::DynamicIndexBufferHandle zero);
    // error (1 invalid index, 2 upstream, 4 malformed range), cells, records, max occupancy
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
private:
    void release();
    std::uint32_t capacity_;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,reduce_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
}
