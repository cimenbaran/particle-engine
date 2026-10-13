#pragma once
#include "core/SparseVoxelStore.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
// Immutable resident snapshot. The source must outlive the snapshot. Creation
// queues an upload; revision() is its content stamp, not a readback acknowledgement.
class GpuVoxelWorld {
public:
    GpuVoxelWorld(const terrain::SparseVoxelStore& source,terrain::Chunk origin);
    ~GpuVoxelWorld();
    GpuVoxelWorld(const GpuVoxelWorld&)=delete;
    GpuVoxelWorld& operator=(const GpuVoxelWorld&)=delete;
    bool isCurrent(const terrain::SparseVoxelStore& source,terrain::Chunk origin) const;
    std::uint64_t revision() const{return revision_;}
    std::uint64_t payloadBytes() const{return bytes_;}
    bgfx::DynamicVertexBufferHandle buffer() const{return buffer_;}
private:
    const terrain::SparseVoxelStore* owner_;
    terrain::Chunk origin_;
    std::uint64_t revision_,bytes_;
    bgfx::DynamicVertexBufferHandle buffer_=BGFX_INVALID_HANDLE;
};
struct VoxelContactContext {
    const GpuVoxelWorld& world;
    const terrain::SparseVoxelStore& source;
    terrain::Chunk origin{};
};
class GpuVoxelWorldSweep {
public:
    GpuVoxelWorldSweep(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuVoxelWorldSweep();
    GpuVoxelWorldSweep(const GpuVoxelWorldSweep&)=delete;
    GpuVoxelWorldSweep& operator=(const GpuVoxelWorldSweep&)=delete;
    // Same segment format as GpuVoxelSweep; maximum displacement .125m per axis.
    // Rejects stale content, different source or mismatched shared origin.
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle segments,
                const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source,terrain::Chunk origin);
    // Three vec4 lanes: normal/time, centre/kind, uint material/voxel/revision low/high.
    // kind0 miss,1 contact,2 overlap,3 invalid,4 swept footprint outside residency.
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    static std::uint64_t payloadBytes(std::uint32_t capacity){return std::uint64_t(capacity?capacity:1)*48;}
private:
    void release();
    std::uint32_t capacity_;
    bgfx::DynamicVertexBufferHandle output_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuVoxelWorldTest(const std::filesystem::path& assets);
}
