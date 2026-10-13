#pragma once
#include "GpuVoxelWorld.hpp"
#include "core/TerrainDeposit.hpp"
namespace pw {
// Private preparation only. No occupancy writes, owner claims or retirement.
class GpuDepositCandidates {
public:
    GpuDepositCandidates(const std::filesystem::path& assets,std::uint32_t capacity,
        std::span<const terrain::DepositMaterial> materials,double tolerance=terrain::maximumConversionError);
    ~GpuDepositCandidates();
    GpuDepositCandidates(const GpuDepositCandidates&)=delete;
    GpuDepositCandidates& operator=(const GpuDepositCandidates&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle particles,bgfx::DynamicVertexBufferHandle history,
        const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source);
    // Two uvec4 rows per slot: signed cell xyz/material, then slot/generation/
    // kind (0 ignored,1 solid blocker,2 eligible owner)/supported-empty-target.
    bgfx::DynamicVertexBufferHandle output()const{return candidates_;}
    // errors (1 identity,2 numeric), eligible owners, ready targets, solid blockers.
    // Any error invalidates the whole proposal; counts are diagnostic only.
    bgfx::DynamicIndexBufferHandle status()const{return status_;}
private:
    void release();
    std::uint32_t capacity_,materials_;
    bgfx::DynamicVertexBufferHandle candidates_=BGFX_INVALID_HANDLE,profiles_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,prepare_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuDepositCandidatesTest(const std::filesystem::path& assets);
}
