#pragma once
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include <array>
namespace pw {
// Position-only common-radius granular Jacobi iteration. Not a full time step:
// prediction, velocity reconstruction and terrain response remain caller work.
class GpuJacobiContacts {
public:
    static constexpr float radius=0.0625f;
    GpuJacobiContacts(const std::filesystem::path& assets,std::uint32_t capacity,
                      float relaxation=0.8f,float maxCorrection=radius*0.5f);
    ~GpuJacobiContacts();
    GpuJacobiContacts(const GpuJacobiContacts&)=delete;
    GpuJacobiContacts& operator=(const GpuJacobiContacts&)=delete;
    // Rebuilds the entire index from this immutable input every iteration.
    // The previous output may be the next input; private state buffers alternate.
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,
                bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount);
    bgfx::DynamicVertexBufferHandle output() const {return output_;}
    // error bits, max overlaps per particle, max candidates, max hash probes.
    // Any error copies input unchanged across the whole iteration.
    bgfx::DynamicIndexBufferHandle status() const {return status_;}
private:
    std::uint32_t capacity_;
    float relaxation_,maxCorrection_;
    GpuSpatialKeys keys_;GpuRadixSort sort_;GpuCellRanges ranges_;GpuCellHash hash_;
    std::array<bgfx::DynamicVertexBufferHandle,2> states_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle corrections_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,solve_=BGFX_INVALID_HANDLE,reduce_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle neighborParams_=BGFX_INVALID_HANDLE,contactParams_=BGFX_INVALID_HANDLE;
    void release();
};
void runGpuJacobiContactsTest(const std::filesystem::path& assets);
}
