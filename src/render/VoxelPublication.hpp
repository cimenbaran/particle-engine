#pragma once
#include "VoxelTerrain.hpp"
#include "PoolDiagnostics.hpp"
namespace pw {
// Immutable CPU/collision/render bundle. References remain valid while held.
class VoxelPublication {
public:
    VoxelPublication(const std::filesystem::path& assets,std::unique_ptr<terrain::SparseVoxelStore> source,
                     terrain::Chunk origin,const MaterialLibrary& materials);
    const terrain::SparseVoxelStore& source() const{return *source_;}
    const GpuVoxelWorld& occupancy() const{return occupancy_;}
    const VoxelTerrain& mesh() const{return mesh_;}
    terrain::Chunk origin() const{return origin_;}
private:
    std::unique_ptr<terrain::SparseVoxelStore> source_;
    terrain::Chunk origin_;
    GpuVoxelWorld occupancy_;
    VoxelTerrain mesh_;
};
using VoxelPublicationPtr=std::shared_ptr<const VoxelPublication>;
// Stages uploads without touching the live publication. This acknowledges the
// occupancy upload only; debris reservation/activation and deposit retirement
// must be coordinated separately before callers commit a physical conversion.
class PendingVoxelPublication {
public:
    enum class State {Prepared,Waiting,Ready,Published,Stale,InvalidAcknowledgement};
    PendingVoxelPublication(const std::filesystem::path& assets,VoxelPublicationPtr base,
        std::span<const terrain::SparseVoxelStore::VoxelEdit> edits,std::uint64_t expectedRevision,
        const MaterialLibrary& materials);
    ~PendingVoxelPublication();
    PendingVoxelPublication(const PendingVoxelPublication&)=delete;
    PendingVoxelPublication& operator=(const PendingVoxelPublication&)=delete;
    // Submit once, after simulation views; diagnostic copy uses view+1.
    void request(std::uint16_t view);
    void poll(std::uint32_t completedFrame);
    // Caller invokes at a tick boundary after any physical conversion gates.
    // No allocation or partial replacement. Retaining base prevents pointer ABA.
    bool publish(VoxelPublicationPtr& live);
    State state() const{return state_;}
    const VoxelPublication& candidate() const{return *candidate_;}
private:
    VoxelPublicationPtr base_,candidate_;
    PoolDiagnostics acknowledgement_;
    State state_=State::Prepared;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
};
void runVoxelPublicationTest(const std::filesystem::path& assets);
}
