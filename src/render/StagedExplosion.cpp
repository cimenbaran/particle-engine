#include "StagedExplosion.hpp"
namespace pw {
StagedExplosion::StagedExplosion(const std::filesystem::path& assets,VoxelProbes& particles,VoxelPublicationPtr terrain,
    const MaterialLibrary& materials,terrain::ExplosionRequest request,std::span<const terrain::ExplosionMaterial> profiles,double existingAllowance)
    :assets_(assets),materials_(materials),request_(request),profiles_(profiles.begin(),profiles.end()),
     owner_(particles.transactionOwner_),expected_(std::move(terrain)),remaining_(request.energyJoules) {
    if(!std::isfinite(existingAllowance) || existingAllowance<0 || existingAllowance>1 || (existingAllowance>0 && request.radius<.0625f))
        throw std::invalid_argument("Invalid existing-grain energy allowance or impulse radius");
    impulseBudget_=request.energyJoules*existingAllowance;impulseDone_=impulseBudget_<1e-6 || request.impulse==0;
    terrain::validateExplosion(request,particles.ticks());terrain::validateExplosionMaterials(profiles);
    if(!particles.hasPool() || particles.transactionPending() || !expected_ || expected_->origin()!=terrain::Chunk{} ||
       profiles.size()!=materials.entries.size() || particles.materialEncoding_!=encodeMaterials(materials) ||
       particles.materialHash_!=materials.contentHash || !expected_->matchesMaterials(materials))
        throw std::invalid_argument("Explosion requires an available matching basin pool and material epoch");
    for(unsigned m=0;m<profiles.size();++m)if(profiles[m].density!=materials.entries[m].gpu.mechanical[0] ||
        profiles[m].solid!=(materials.entries[m].gpu.identity[1]==0) ||
        profiles[m].resistance!=materials.entries[m].fractureResistance ||
        profiles[m].fuel!=(materials.entries[m].gpu.thermal[3]>0?1.f:0.f))
        throw std::invalid_argument("Explosion energy profiles must match physical materials");
}
void StagedExplosion::cancel() {if(!active())return;impulse_.reset();pending_.reset();expected_.reset();state_=State::Cancelled;}
bool StagedExplosion::pump(std::uint32_t completedFrame,VoxelProbes& particles,VoxelPublicationPtr& terrain,
                           std::uint16_t prepareView,std::uint16_t commitView) {
    if(!active())return false;
    if(prepareView>=bgfx::getCaps()->limits.maxViews-1 || commitView>=prepareView)
        throw std::invalid_argument("Explosion requires an early commit and later preparation/copy views");
    if(owner_.expired() || owner_.lock()!=particles.transactionOwner_ || terrain!=expected_ ||
       (attempted_ && particles.ticks()<lastAttempt_)) {
        impulse_.reset();pending_.reset();expected_.reset();state_=State::Stale;return false;
    }
    if(impulse_) {
        impulse_->poll(completedFrame);
        if(impulse_->state()==PendingBlastImpulse::State::Waiting)return false;
        if(impulse_->state()==PendingBlastImpulse::State::Ready) {
            const auto charge=impulse_->energyJoules(),remaining=terrain::explosionEnergyRemainder(remaining_,charge);
            const auto accepted=impulse_->accepted();
            if(!impulse_->publish(commitView,particles,terrain)){impulse_.reset();expected_.reset();state_=State::Stale;return false;}
            remaining_=remaining;impulseCharge_=charge;impulsed_=accepted;impulseDone_=true;
            if(accepted)armCosmetics();
            impulse_.reset();state_=State::WaitingForTick;errors_=0;return true;
        }
        errors_=impulse_->errors();impulse_.reset();expected_.reset();state_=State::Faulted;return false;
    }
    if(pending_) {
        pending_->poll(completedFrame);
        if(pending_->state()==PendingDebrisTransaction::State::Waiting)return false;
        if(pending_->state()==PendingDebrisTransaction::State::Ready) {
            // Validate all arithmetic before the nonthrowing joint adoption.
            const auto remaining=terrain::explosionEnergyRemainder(remaining_,pendingCharge_);
            if(!pending_->publish(commitView,particles,terrain)){pending_.reset();expected_.reset();state_=State::Stale;return false;}
            armCosmetics();remaining_=remaining;expected_=terrain;++batches_;removed_+=pendingVoxels_;grains_+=pendingGrains_;
            pending_.reset();state_=State::WaitingForTick;errors_=0;
            return true; // Never start another batch in the publication call.
        }
        errors_=pending_->errors();pending_.reset();
        state_=((errors_&(1u|8u)) && !(errors_&~(1u|8u|64u)))?State::Deferred:State::Faulted;
        if(state_==State::Faulted)expected_.reset();
        return false;
    }
    if(particles.fault()){expected_.reset();state_=State::Faulted;return false;}
    if(particles.transactionPending() || particles.pendingCommands())return false;
    if(attempted_ && particles.ticks()-lastAttempt_<(state_==State::Deferred?retryTicks:1))return false;
    lastAttempt_=particles.ticks();attempted_=true;
    if(!impulseDone_) {
        BlastImpulseRequest request{request_.centre,request_.radius,request_.impulse,impulseBudget_};
        auto transaction=std::make_unique<PendingBlastImpulse>(assets_,particles,terrain,materials_,request);
        transaction->submit(prepareView,particles);impulse_=std::move(transaction);state_=State::WaitingForGpu;return false;
    }
    auto batch=request_;batch.tick=particles.ticks();batch.energyJoules=remaining_;
    const auto plan=terrain::planExplosion(terrain->source(),batch,batch.tick,profiles_);
    eligible_=plan.eligible;
    using Result=terrain::ExplosionPlan::State;
    if(plan.state!=Result::Ready) {
        if(plan.state==Result::NoSpace)state_=State::NoSpace;
        else {state_=State::Complete;completion_=plan.state==Result::Empty?Completion::Empty:plan.state==Result::InsufficientEnergy?Completion::Energy:Completion::Quota;}
        expected_.reset();return false;
    }
    auto transaction=std::make_unique<PendingDebrisTransaction>(assets_,particles,terrain,plan.removals,plan.seeds,materials_);
    transaction->submit(prepareView,particles);
    pendingCharge_=plan.reservedJoules;pendingVoxels_=std::uint32_t(plan.removals.size());pendingGrains_=std::uint32_t(plan.seeds.size());
    pending_=std::move(transaction);state_=State::WaitingForGpu;
    return false;
}
}
