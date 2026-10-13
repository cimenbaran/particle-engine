#include "PendingDebrisTransaction.hpp"
#include "Shaders.hpp"
namespace pw {
namespace {
std::vector<DebrisMassTotals> planMass(const VoxelPublication& base,
    std::span<const terrain::SparseVoxelStore::VoxelEdit> removals,std::span<const DebrisSeed> seeds,
    const MaterialLibrary& materials,const std::vector<DebrisMassTotals>& previous,double tolerance) {
    terrain::mass_detail::tolerance(tolerance);
    if(removals.empty() || removals.size()>4096 || seeds.size()>4096)throw std::invalid_argument("Debris transaction requires 1..4096 removals and at most 4096 seeds");
    std::vector<std::uint32_t> voxels(materials.entries.size()),grains(voxels.size());
    for(const auto& edit:removals) {
        const auto old=base.source().materialAt(edit.address);
        if(edit.material!=terrain::SparseVoxelStore::empty || !old || *old==terrain::SparseVoxelStore::empty || *old>=voxels.size())throw std::invalid_argument("Debris transaction requires occupied voxel removals");
        ++voxels[*old];
    }
    for(const auto& seed:seeds){if(seed.material>=grains.size())throw std::invalid_argument("Invalid debris material");++grains[seed.material];}
    auto ledger=previous;
    for(unsigned material=0;material<voxels.size();++material) {
        if(!voxels[material] && !grains[material])continue;
        const auto& gpu=materials.entries[material].gpu;
        if(gpu.identity[0]!=material || gpu.identity[1]!=0)throw std::invalid_argument("Debris conversion requires stable solid materials");
        const auto balance=terrain::planVoxelDebris(voxels[material],gpu.mechanical[0],tolerance);
        if(!balance.admissible || balance.grains!=grains[material])throw std::invalid_argument("Debris seed counts do not match per-material mass plans");
        auto& total=ledger[material];
        if(total.removedVoxels>UINT64_MAX-balance.voxels || total.activatedGrains>UINT64_MAX-balance.grains)throw std::overflow_error("Debris mass count exhausted");
        total.removedVoxels+=balance.voxels;total.activatedGrains+=balance.grains;
        total.sourceKilograms+=balance.sourceKilograms;total.destinationKilograms+=balance.destinationKilograms;total.quantizationKilograms+=balance.quantizationKilograms;
        if(!std::isfinite(total.sourceKilograms) || !std::isfinite(total.destinationKilograms) || !std::isfinite(total.quantizationKilograms))throw std::overflow_error("Debris mass total exhausted");
    }
    return ledger;
}
}
PendingDebrisTransaction::PendingDebrisTransaction(const std::filesystem::path& assets,VoxelProbes& particles,
    VoxelPublicationPtr base,std::span<const terrain::SparseVoxelStore::VoxelEdit> removals,
    std::span<const DebrisSeed> seeds,const MaterialLibrary& materials,double tolerance)
    :base_(std::move(base)) {
    if(!particles.hasPool() || particles.transactionPending() || !base_ || base_->origin()!=terrain::Chunk{} ||
        materials.contentHash!=particles.materialHash_ || materials.contentHash!=base_->materialHash() || materials.entries.size()!=particles.materials_ ||
        particles.materialEncoding_!=encodeMaterials(materials) || !base_->matchesMaterials(materials))
        throw std::invalid_argument("Debris conversion requires an unfrozen basin pool and matching terrain/material epoch");
    ledger_=planMass(*base_,removals,seeds,materials,particles.debrisMass_,tolerance);
    // Candidate creation validates duplicate destinations and allocates all
    // occupancy/render resources before the live pool is frozen.
    terrain_=std::make_unique<PendingVoxelPublication>(assets,base_,removals,base_->source().revision(),materials);
    reservation_=std::make_unique<GpuDebrisReservation>(assets,particles.capacity_);
    activation_=std::make_unique<GpuDebrisActivation>(assets,particles.capacity_,materials,seeds);
    clearance_=std::make_unique<GpuDebrisClearance>(assets,particles.capacity_);
    try {
        acknowledgement_=std::make_unique<PoolDiagnostics>(assets);
        acknowledge_=loadCompute(assets,"cs_debris_ack");
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        if(!bgfx::isValid(status_))throw std::runtime_error("Debris acknowledgement allocation failed");
        lease_=std::make_shared<const int>(0);
    }catch(...){release();throw;}
    owner_=particles.transactionOwner_;source_=particles.state();tick_=particles.ticks_;materialHash_=particles.materialHash_;count_=std::uint32_t(seeds.size());
    particles.transactionLease_=lease_;
}
void PendingDebrisTransaction::release() {
    // Readback destinations and the freeze must survive execution, including
    // abandonment while Waiting. Terrain's acknowledgement drains separately.
    acknowledgement_.reset();
    if(bgfx::isValid(acknowledge_))bgfx::destroy(acknowledge_);
    if(bgfx::isValid(status_))bgfx::destroy(status_);
}
PendingDebrisTransaction::~PendingDebrisTransaction(){release();}
bool PendingDebrisTransaction::matches(const VoxelProbes& particles) const {
    return !owner_.expired() && owner_.lock()==particles.transactionOwner_ &&
        particles.transactionLease_.lock()==lease_ && particles.state().idx==source_.idx &&
        particles.ticks_==tick_ && particles.materialHash_==materialHash_;
}
void PendingDebrisTransaction::submit(std::uint16_t view,VoxelProbes& particles) {
    if(state_!=State::Prepared)throw std::logic_error("Debris transaction already submitted");
    if(view>=bgfx::getCaps()->limits.maxViews-1)throw std::invalid_argument("Debris transaction requires a following copy view");
    if(!matches(particles))throw std::invalid_argument("Stale debris particle owner");
    const auto& candidate=terrain_->candidate();const VoxelContactContext context{candidate.occupancy(),candidate.source(),candidate.origin()};
    reservation_->submit(view,source_,count_);activation_->submit(view,*reservation_);
    clearance_->submit(view,*activation_,particles.materialBuffer_,particles.materials_,context);
    terrain_->request(view);
    bgfx::setBuffer(0,particles.latched_,bgfx::Access::Read);bgfx::setBuffer(1,clearance_->status(),bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Write);bgfx::dispatch(view,acknowledge_,1);
    if(!acknowledgement_->request(view,status_,tick_))throw std::logic_error("Debris acknowledgement unavailable");
    state_=State::Waiting;
}
void PendingDebrisTransaction::poll(std::uint32_t frame) {
    if(state_!=State::Waiting)return;
    terrain_->poll(frame);acknowledgement_->poll(frame);
    const auto& ack=acknowledgement_->latest();
    if(!ack.valid || terrain_->state()==PendingVoxelPublication::State::Waiting)return;
    errors_=ack.active;
    if(ack.free!=count_ || ack.reserved!=count_ || ack.rejected!=0)errors_|=64;
    if(terrain_->state()!=PendingVoxelPublication::State::Ready)errors_|=128;
    state_=errors_?State::Rejected:State::Ready;
    if(errors_)lease_.reset();
}
bool PendingDebrisTransaction::publish(std::uint16_t view,VoxelProbes& particles,VoxelPublicationPtr& terrain) {
    if(state_!=State::Ready)return false;
    if(view>=bgfx::getCaps()->limits.maxViews)throw std::invalid_argument("Invalid debris commit view");
    if(!matches(particles) || terrain!=base_) {state_=State::Stale;lease_.reset();return false;}
    // All allocations and asynchronous gates are complete. From here adoption
    // uses pointer/handle/vector swaps; no partial CPU publication can throw.
    if(!terrain_->publish(terrain)){state_=State::Stale;lease_.reset();return false;}
    std::swap(particles.states_[particles.current_],activation_->output_);
    particles.debrisMass_.swap(ledger_);
    particles.pool_->rebuild(view,particles.state());
    particles.settling_->reset(view);
    particles.poolDiagnostics_->beginEpoch();particles.settlingDiagnostics_->beginEpoch();
    state_=State::Published;lease_.reset();return true;
}
}
