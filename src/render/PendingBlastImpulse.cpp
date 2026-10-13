#include "PendingBlastImpulse.hpp"
#include "Shaders.hpp"
namespace pw {
PendingBlastImpulse::PendingBlastImpulse(const std::filesystem::path& assets,VoxelProbes& particles,VoxelPublicationPtr terrain,
    const MaterialLibrary& materials,BlastImpulseRequest request):terrain_(std::move(terrain)) {
    if(!particles.hasPool() || particles.transactionPending() || !terrain_ || terrain_->origin()!=terrain::Chunk{} ||
       materials.contentHash!=particles.materialHash_ || particles.materialEncoding_!=encodeMaterials(materials) || !terrain_->matchesMaterials(materials))
        throw std::invalid_argument("Blast impulse requires an available matching basin pool and material epoch");
    impulse_=std::make_unique<GpuBlastImpulse>(assets,particles.capacity_,request);
    try {
        acknowledgement_=std::make_unique<PoolDiagnostics>(assets);
        acknowledge_=loadCompute(assets,"cs_debris_ack"); // Shared health + four-word acknowledgement gate.
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        if(!bgfx::isValid(status_))throw std::runtime_error("Blast acknowledgement allocation failed");
        lease_=std::make_shared<const int>(0);
    }catch(...){release();throw;}
    owner_=particles.transactionOwner_;tick_=particles.ticks_;source_=particles.state();capacity_=particles.capacity_;materialHash_=particles.materialHash_;
    particles.transactionLease_=lease_;
}
void PendingBlastImpulse::release() {
    acknowledgement_.reset(); // Drain before releasing destinations or freeze.
    if(bgfx::isValid(acknowledge_))bgfx::destroy(acknowledge_);
    if(bgfx::isValid(status_))bgfx::destroy(status_);
}
PendingBlastImpulse::~PendingBlastImpulse(){release();}
bool PendingBlastImpulse::matches(const VoxelProbes& particles) const {
    return !owner_.expired() && owner_.lock()==particles.transactionOwner_ && particles.transactionLease_.lock()==lease_ &&
        particles.state().idx==source_.idx && particles.ticks_==tick_ && particles.materialHash_==materialHash_;
}
void PendingBlastImpulse::submit(std::uint16_t view,VoxelProbes& particles) {
    if(state_!=State::Prepared)throw std::logic_error("Blast impulse already submitted");
    if(view>=bgfx::getCaps()->limits.maxViews-1 || !matches(particles))throw std::invalid_argument("Invalid blast view or particle owner");
    impulse_->submit(view,source_,particles.materialBuffer_,particles.materials_);
    bgfx::setBuffer(0,particles.latched_,bgfx::Access::Read);bgfx::setBuffer(1,impulse_->status(),bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,acknowledge_,1);
    if(!acknowledgement_->request(view,status_,tick_))throw std::logic_error("Blast acknowledgement unavailable");
    state_=State::Waiting;
}
void PendingBlastImpulse::poll(std::uint32_t frame) {
    if(state_!=State::Waiting)return;
    acknowledgement_->poll(frame);const auto& ack=acknowledgement_->latest();if(!ack.valid)return;
    errors_=ack.active;
    if(ack.free>capacity_ || ack.reserved>ack.free || ack.rejected>impulse_->ticketBudget() ||
       ((ack.reserved==0)!=(ack.rejected==0)))errors_|=64;
    if(errors_){state_=State::Rejected;lease_.reset();return;}
    accepted_=ack.reserved;energy_=double(ack.rejected)*impulse_->joulesPerTicket();state_=State::Ready;
}
bool PendingBlastImpulse::publish(std::uint16_t view,VoxelProbes& particles,const VoxelPublicationPtr& terrain) {
    if(state_!=State::Ready)return false;
    if(view>=bgfx::getCaps()->limits.maxViews)throw std::invalid_argument("Invalid blast publication view");
    if(!matches(particles) || terrain!=terrain_){state_=State::Stale;energy_=0;accepted_=0;lease_.reset();return false;}
    if(accepted_) {
        std::swap(particles.states_[particles.current_],impulse_->output_);
        particles.pool_->rebuild(view,particles.state());particles.settling_->reset(view);
        particles.poolDiagnostics_->beginEpoch();particles.settlingDiagnostics_->beginEpoch();
    }
    state_=State::Published;lease_.reset();return true;
}
}
