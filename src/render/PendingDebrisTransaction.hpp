#pragma once
#include "VoxelProbes.hpp"
#include "VoxelPublication.hpp"
#include "GpuDebrisClearance.hpp"
#include "core/TerrainMass.hpp"
namespace pw {
// A removal-only physical conversion. Freeze the command pool while preparing;
// publish its complete private particle snapshot and terrain at one boundary.
class PendingDebrisTransaction {
public:
    enum class State {Prepared,Waiting,Ready,Published,Rejected,Stale};
    PendingDebrisTransaction(const std::filesystem::path& assets,VoxelProbes& particles,
        VoxelPublicationPtr base,std::span<const terrain::SparseVoxelStore::VoxelEdit> removals,
        std::span<const DebrisSeed> seeds,const MaterialLibrary& materials,
        double tolerance=terrain::maximumConversionError);
    ~PendingDebrisTransaction();
    PendingDebrisTransaction(const PendingDebrisTransaction&)=delete;
    PendingDebrisTransaction& operator=(const PendingDebrisTransaction&)=delete;
    // Same sequential view as, or later than, all base simulation submissions.
    // Copy uses view+1. No raw writes to exposed live handles while frozen.
    void submit(std::uint16_t view,VoxelProbes& particles);
    void poll(std::uint32_t completedFrame);
    // Invoke at a tick boundary, before subsequent simulation/render views.
    // Neither pre-ACK, rejected, stale nor repeated commits change live state.
    bool publish(std::uint16_t view,VoxelProbes& particles,VoxelPublicationPtr& terrain);
    State state() const{return state_;}
    std::uint32_t errors() const{return errors_;}
    const VoxelPublication& candidate() const{return terrain_->candidate();}
private:
    bool matches(const VoxelProbes& particles) const;
    void release();
    // Declared first: lease survives all queued work/readback teardown.
    std::shared_ptr<const int> lease_;
    std::weak_ptr<const int> owner_;
    VoxelPublicationPtr base_;
    std::uint64_t tick_=0,materialHash_=0;
    bgfx::DynamicVertexBufferHandle source_=BGFX_INVALID_HANDLE;
    std::uint32_t count_=0,errors_=0;
    State state_=State::Prepared;
    std::vector<DebrisMassTotals> ledger_;
    std::unique_ptr<PendingVoxelPublication> terrain_;
    std::unique_ptr<GpuDebrisReservation> reservation_;
    std::unique_ptr<GpuDebrisActivation> activation_;
    std::unique_ptr<GpuDebrisClearance> clearance_;
    bgfx::ProgramHandle acknowledge_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    std::unique_ptr<PoolDiagnostics> acknowledgement_;
};
void runPendingDebrisTransactionTest(const std::filesystem::path& assets);
}
