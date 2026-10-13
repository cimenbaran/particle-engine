#include "VoxelPublication.hpp"
#include "Shaders.hpp"
namespace pw {
namespace {
std::unique_ptr<terrain::SparseVoxelStore> checked(std::unique_ptr<terrain::SparseVoxelStore> source,const MaterialLibrary& materials) {
    if(!source)throw std::invalid_argument("Terrain publication requires a source");
    if(source->layout().chunkCount()>256)throw std::invalid_argument("Terrain publication exceeds 256 chunks");
    std::array<std::uint32_t,terrain::voxelsPerChunk> chunk;
    for(unsigned slot=0;slot<source->layout().chunkCount();++slot) {
        source->copyChunk(*source->layout().decode(slot),chunk);
        for(auto material:chunk)if(material!=terrain::SparseVoxelStore::empty &&
            (material>=materials.entries.size() || materials.entries[material].gpu.identity[1]!=0))
            throw std::invalid_argument("Terrain publication requires solid materials, including hidden voxels");
    }
    return source;
}
VoxelPublicationPtr prepare(const std::filesystem::path& assets,const VoxelPublicationPtr& base,
    std::span<const terrain::SparseVoxelStore::VoxelEdit> edits,std::uint64_t revision,const MaterialLibrary& materials) {
    if(!base)throw std::invalid_argument("Terrain edit requires a base publication");
    auto copy=base->source().clone();
    const auto result=copy->setMaterials(edits,revision);
    if(result==terrain::SparseVoxelStore::Edit::Stale)throw std::invalid_argument("Stale terrain edit revision");
    if(result==terrain::SparseVoxelStore::Edit::Unchanged)return base;
    return std::make_shared<VoxelPublication>(assets,std::move(copy),base->origin(),materials);
}
}
VoxelPublication::VoxelPublication(const std::filesystem::path& assets,std::unique_ptr<terrain::SparseVoxelStore> source,
    terrain::Chunk origin,const MaterialLibrary& materials)
    :source_(checked(std::move(source),materials)),origin_(origin),materialHash_(materials.contentHash),materialEncoding_(encodeMaterials(materials)),occupancy_(*source_,origin),mesh_(assets,*source_,occupancy_,origin,materials) {}
PendingVoxelPublication::PendingVoxelPublication(const std::filesystem::path& assets,VoxelPublicationPtr base,
    std::span<const terrain::SparseVoxelStore::VoxelEdit> edits,std::uint64_t revision,const MaterialLibrary& materials)
    :base_(std::move(base)),candidate_(prepare(assets,base_,edits,revision,materials)),acknowledgement_(assets) {
    program_=loadCompute(assets,"cs_voxel_ack");
    status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
    if(!bgfx::isValid(status_)){bgfx::destroy(program_);throw std::runtime_error("Terrain acknowledgement allocation failed");}
}
PendingVoxelPublication::~PendingVoxelPublication() {
    if(bgfx::isValid(program_))bgfx::destroy(program_);
    if(bgfx::isValid(status_))bgfx::destroy(status_);
    // acknowledgement_ retains readback destinations until completion; both
    // immutable publications outlive its destructor and queued GPU use.
}
void PendingVoxelPublication::request(std::uint16_t view) {
    if(state_!=State::Prepared)throw std::logic_error("Terrain acknowledgement already requested");
    if(view>=bgfx::getCaps()->limits.maxViews-1)throw std::invalid_argument("Terrain acknowledgement needs a following copy view");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,candidate_->occupancy().buffer(),bgfx::Access::Read);
    bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,program_,1);
    if(!acknowledgement_.request(view,status_,candidate_->source().revision()))throw std::logic_error("Terrain acknowledgement ring unavailable");
    state_=State::Waiting;
}
void PendingVoxelPublication::poll(std::uint32_t frame) {
    acknowledgement_.poll(frame);if(state_!=State::Waiting)return;
    const auto& ack=acknowledgement_.latest();if(!ack.valid)return;
    const auto& source=candidate_->source();const auto revision=source.revision();
    state_=ack.active==std::uint32_t(revision) && ack.free==std::uint32_t(revision>>32) &&
        ack.reserved==source.layout().chunkCount() && ack.rejected==source.layout().voxelCount()?State::Ready:State::InvalidAcknowledgement;
}
bool PendingVoxelPublication::publish(VoxelPublicationPtr& live) {
    if(state_==State::Published || state_==State::Stale || state_==State::InvalidAcknowledgement)return false;
    if(live!=base_){state_=State::Stale;return false;}
    if(state_!=State::Ready)return false;
    live=candidate_;state_=State::Published;return true;
}
}
