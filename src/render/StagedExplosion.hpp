#pragma once
#include "PendingDebrisTransaction.hpp"
#include "PendingBlastImpulse.hpp"
#include "core/TerrainExplosion.hpp"
namespace pw {
// Simulation-thread coordinator. Call pump at a frame/tick boundary before
// physics or rendering. Publication and energy charging happen together here.
class StagedExplosion {
public:
    enum class State {WaitingForTick,WaitingForGpu,Deferred,Complete,NoSpace,Cancelled,Stale,Faulted};
    enum class Completion {None,Empty,Energy,Quota};
    StagedExplosion(const std::filesystem::path& assets,VoxelProbes& particles,VoxelPublicationPtr terrain,
                    const MaterialLibrary& materials,terrain::ExplosionRequest request,
                    std::span<const terrain::ExplosionMaterial> profiles,double existingAllowance=0);
    // True only if an impulse or terrain batch was published; refresh any cached world aliases.
    bool pump(std::uint32_t completedFrame,VoxelProbes& particles,VoxelPublicationPtr& terrain,
              std::uint16_t prepareView=8,std::uint16_t commitView=0);
    void cancel();
    bool active() const{return state_==State::WaitingForTick || state_==State::WaitingForGpu || state_==State::Deferred;}
    State state() const{return state_;}
    Completion completion() const{return completion_;}
    double remainingJoules() const{return remaining_;}
    double chargedJoules() const{return request_.energyJoules-remaining_;}
    std::uint64_t batches() const{return batches_;}
    std::uint64_t removedVoxels() const{return removed_;}
    std::uint64_t activatedGrains() const{return grains_;}
    std::uint32_t impulsedParticles() const{return impulsed_;}
    double impulseJoules() const{return impulseCharge_;}
    bool applyingImpulse() const{return bool(impulse_);}
    std::uint32_t errors() const{return errors_;}
    std::uint32_t eligibleVoxels() const{return eligible_;}
    static constexpr std::uint64_t retryTicks=30;
private:
    std::filesystem::path assets_;
    MaterialLibrary materials_;
    terrain::ExplosionRequest request_;
    std::vector<terrain::ExplosionMaterial> profiles_;
    std::weak_ptr<const int> owner_;
    VoxelPublicationPtr expected_;
    State state_=State::WaitingForTick;
    Completion completion_=Completion::None;
    double remaining_=0,pendingCharge_=0,impulseBudget_=0,impulseCharge_=0;
    std::uint32_t impulsed_=0;
    bool impulseDone_=true;
    std::uint64_t batches_=0,removed_=0,grains_=0,lastAttempt_=0;
    std::uint32_t pendingVoxels_=0,pendingGrains_=0,errors_=0,eligible_=0;
    bool attempted_=false;
    std::unique_ptr<PendingDebrisTransaction> pending_;
    std::unique_ptr<PendingBlastImpulse> impulse_;
};
void runStagedExplosionTest(const std::filesystem::path& assets);
}
