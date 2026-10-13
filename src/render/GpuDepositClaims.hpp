#pragma once
#include "GpuDepositCandidates.hpp"
namespace pw {
// Private, bounded greedy arbitration and retirement candidate. No live publication.
class GpuDepositClaims {
public:
    GpuDepositClaims(const std::filesystem::path& assets,std::uint32_t capacity,
        std::span<const terrain::DepositMaterial> materials,std::uint32_t quota=32,
        double tolerance=terrain::maximumConversionError);
    ~GpuDepositClaims();
    GpuDepositClaims(const GpuDepositClaims&)=delete;
    GpuDepositClaims& operator=(const GpuDepositClaims&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle particles,bgfx::DynamicVertexBufferHandle history,
        const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source);
    // Two uvec4 per claim: signed target xyz/material, then slotA/genA/slotB/genB.
    bgfx::DynamicVertexBufferHandle claims()const{return claims_;}
    // One uint per source slot: zero unclaimed, otherwise one-based claim index.
    bgfx::DynamicIndexBufferHandle owners()const{return owners_;}
    // errors, accepted claims, claimed grains, clearance-deferred proposals.
    bgfx::DynamicIndexBufferHandle status()const{return status_;}
    // Private copy: only claimed owners have their alive flag cleared.
    bgfx::DynamicVertexBufferHandle retired()const{return retired_;}
    // errors (upstream, 64 ownership/count, 128 identity), claims, grains, zero.
    bgfx::DynamicIndexBufferHandle retirementStatus()const{return retirementStatus_;}
private:
    void release();
    GpuDepositCandidates candidates_;
    std::uint32_t capacity_,quota_;
    bgfx::DynamicVertexBufferHandle claims_=BGFX_INVALID_HANDLE,retired_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle owners_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE,retirementStatus_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE,retireProgram_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuDepositClaimsTest(const std::filesystem::path& assets);
}
