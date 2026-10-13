#pragma once
#include "GpuVoxelWorld.hpp"
#include "assets/Materials.hpp"
namespace pw {
class VoxelTerrain {
public:
    VoxelTerrain(const std::filesystem::path& assets,const terrain::SparseVoxelStore& source,
                 const GpuVoxelWorld& world,terrain::Chunk origin,const MaterialLibrary& materials);
    ~VoxelTerrain();
    VoxelTerrain(const VoxelTerrain&)=delete;VoxelTerrain& operator=(const VoxelTerrain&)=delete;
    void render(std::uint16_t view,const terrain::SparseVoxelStore& source,const GpuVoxelWorld& world,terrain::Chunk origin) const;
    std::uint32_t faceCount() const{return faces_;}
    std::uint64_t payloadBytes() const{return std::uint64_t(faces_)*88;}
private:
    void release();
    const terrain::SparseVoxelStore* owner_;
    terrain::Chunk origin_;
    std::uint64_t revision_;
    std::uint32_t faces_=0;
    bgfx::VertexBufferHandle vertices_=BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
};
void runVoxelTerrainTest(const std::filesystem::path& assets);
}
