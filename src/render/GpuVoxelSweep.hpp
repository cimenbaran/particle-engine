#pragma once
#include <bgfx/bgfx.h>
#include <array>
#include <filesystem>
namespace pw {
// Independent single-voxel geometry queries. Does not mutate particle state,
// traverse occupancy, apply a response or publish terrain revisions.
class GpuVoxelSweep {
public:
    GpuVoxelSweep(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuVoxelSweep();
    GpuVoxelSweep(const GpuVoxelSweep&)=delete;
    GpuVoxelSweep& operator=(const GpuVoxelSweep&)=delete;
    // Input: two vec4 lanes/slot (start xyz,radius=.0625; end xyz,reserved=0).
    // Coordinates must be finite within +/-64m, relative to the shared origin.
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle segments,std::array<float,3> lower);
    // Output: normal xyz,time; centre xyz,kind as uint bits.
    // kind0 miss,1 closed contact,2 initial overlap,3 invalid input/numerics.
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    static std::uint64_t payloadBytes(std::uint32_t capacity){return std::uint64_t(capacity?capacity:1)*32;}
private:
    void release();
    std::uint32_t capacity_;
    bgfx::DynamicVertexBufferHandle output_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuVoxelSweepTest(const std::filesystem::path& assets);
}
