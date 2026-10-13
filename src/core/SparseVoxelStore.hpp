#pragma once
#include "VoxelCoordinates.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <span>
#include <set>
#include <vector>

namespace pw::terrain {
// Single-threaded authoritative occupancy. Empty is separate from all 65536
// supported material IDs. Revisions describe CPU state, not GPU publication.
class SparseVoxelStore {
public:
    static constexpr std::uint32_t empty=UINT32_MAX;
    struct Revisions {
        std::uint64_t content=0,surface=0;
        bool operator==(const Revisions&) const=default;
    };
    enum class Edit {Changed,Unchanged,Stale};
    struct VoxelEdit {VoxelAddress address;std::uint32_t material;};
    SparseVoxelStore(ResidentChunkLayout layout,std::uint32_t materials)
        :layout_(layout),materials_(materials) {
        if(!materials || materials>65536)throw std::invalid_argument("Invalid voxel material count");
    }
    SparseVoxelStore(const SparseVoxelStore&)=delete;
    SparseVoxelStore& operator=(const SparseVoxelStore&)=delete;
    const ResidentChunkLayout& layout() const{return layout_;}
    std::uint64_t revision() const{return revision_;}
    std::uint64_t occupiedVoxelCount() const{return occupied_;}
    std::uint32_t allocatedChunkCount() const{return allocated_;}
    std::size_t revisionRecordCount() const{return chunks_.size();}
    std::uint64_t occupancyPayloadBytes() const{return std::uint64_t(allocated_)*sizeof(Voxels);}
    std::optional<std::uint32_t> materialAt(VoxelAddress address) const {
        auto slot=layout_.index(address.chunk),local=localIndex(address.local);
        if(!slot || !local)return {};
        auto found=chunks_.find(*slot);
        return found==chunks_.end() || !found->second.voxels?empty:(*found->second.voxels)[*local];
    }
    std::optional<Revisions> revisions(Chunk chunk) const {
        auto slot=layout_.index(chunk);if(!slot)return {};
        auto found=chunks_.find(*slot);
        return found==chunks_.end()?Revisions{}:found->second.revisions;
    }
    // No allocation; invalid chunks leave the caller's destination untouched.
    std::optional<Revisions> copyChunk(Chunk chunk,std::span<std::uint32_t,voxelsPerChunk> destination) const {
        auto slot=layout_.index(chunk);if(!slot)return {};
        auto found=chunks_.find(*slot);
        if(found==chunks_.end() || !found->second.voxels)std::fill(destination.begin(),destination.end(),empty);
        else std::copy(found->second.voxels->begin(),found->second.voxels->end(),destination.begin());
        return revisions(chunk);
    }
    // Invalid address/material throws before checking the expected revision.
    // No-op edits do not advance revisions. Stale callers never mutate state.
    Edit setMaterial(VoxelAddress address,std::uint32_t material,std::uint64_t expectedRevision) {
        auto slot=layout_.index(address.chunk),local=localIndex(address.local);
        if(!slot || !local || (material!=empty && material>=materials_))
            throw std::invalid_argument("Invalid voxel edit");
        if(expectedRevision!=revision_)return Edit::Stale;
        const auto previous=*materialAt(address);
        if(previous==material)return Edit::Unchanged;
        if(revision_==UINT64_MAX)throw std::overflow_error("Voxel revision exhausted");
        // At a corner only three face-sharing neighbors need surface invalidation.
        std::array<std::uint32_t,4> affected{*slot};unsigned count=1;
        for(unsigned axis=0;axis<3;++axis) {
            const int step=address.local[axis]==0?-1:address.local[axis]==31?1:0;
            if(!step)continue;
            const auto coordinate=std::int64_t(address.chunk[axis])+step;
            if(coordinate<INT32_MIN || coordinate>INT32_MAX)continue;
            auto neighbor=address.chunk;neighbor[axis]=std::int32_t(coordinate);
            if(auto next=layout_.index(neighbor))affected[count++]=*next;
        }
        auto found=chunks_.find(*slot);
        std::unique_ptr<Voxels> newVoxels;
        if(material!=empty && (found==chunks_.end() || !found->second.voxels)) {
            newVoxels=std::make_unique<Voxels>();newVoxels->fill(empty);
        }
        // Complete allocations before committing occupancy or revision changes.
        // Allocation failure may leave empty revision records, but no edits.
        for(unsigned i=0;i<count;++i)chunks_.try_emplace(affected[i]);
        auto& own=chunks_.at(*slot);
        if(newVoxels){own.voxels=std::move(newVoxels);++allocated_;}
        (*own.voxels)[*local]=material;
        if(previous==empty){++own.occupied;++occupied_;}
        if(material==empty){--own.occupied;--occupied_;}
        if(!own.occupied){own.voxels.reset();--allocated_;}
        ++revision_;own.revisions.content=revision_;
        for(unsigned i=0;i<count;++i)chunks_.at(affected[i]).revisions.surface=revision_;
        return Edit::Changed;
    }
    // Unique destinations, one world revision, and all-or-nothing CPU content.
    // Validation (including duplicate detection) precedes stale-revision checks.
    // This does not reserve debris or publish a GPU transaction.
    Edit setMaterials(std::span<const VoxelEdit> edits,std::uint64_t expectedRevision) {
        struct Write {std::uint32_t slot,local,material;VoxelAddress address;};
        std::vector<Write> writes;writes.reserve(edits.size());
        for(const auto& edit:edits) {
            auto slot=layout_.index(edit.address.chunk),local=localIndex(edit.address.local);
            if(!slot || !local || (edit.material!=empty && edit.material>=materials_))
                throw std::invalid_argument("Invalid voxel batch edit");
            writes.push_back({*slot,*local,edit.material,edit.address});
        }
        std::sort(writes.begin(),writes.end(),[](const Write& a,const Write& b){
            return a.slot<b.slot || (a.slot==b.slot && a.local<b.local);
        });
        for(std::size_t i=1;i<writes.size();++i)
            if(writes[i-1].slot==writes[i].slot && writes[i-1].local==writes[i].local)
                throw std::invalid_argument("Duplicate voxel batch destination");
        if(expectedRevision!=revision_)return Edit::Stale;
        std::map<std::uint32_t,Entry> staged;
        std::set<std::uint32_t> affected;
        for(const auto& write:writes) {
            const auto previous=*materialAt(write.address);
            if(previous==write.material)continue;
            if(revision_==UINT64_MAX)throw std::overflow_error("Voxel revision exhausted");
            auto [prepared,inserted]=staged.try_emplace(write.slot);
            auto& entry=prepared->second;
            if(inserted) {
                auto old=chunks_.find(write.slot);
                entry.voxels=std::make_unique<Voxels>();
                if(old!=chunks_.end() && old->second.voxels) {
                    *entry.voxels=*old->second.voxels;entry.occupied=old->second.occupied;
                } else entry.voxels->fill(empty);
            }
            (*entry.voxels)[write.local]=write.material;
            if(previous==empty)++entry.occupied;
            if(write.material==empty)--entry.occupied;
            affected.insert(write.slot);
            for(unsigned axis=0;axis<3;++axis) {
                const int step=write.address.local[axis]==0?-1:write.address.local[axis]==31?1:0;
                if(!step)continue;
                const auto coordinate=std::int64_t(write.address.chunk[axis])+step;
                if(coordinate<INT32_MIN || coordinate>INT32_MAX)continue;
                auto neighbor=write.address.chunk;neighbor[axis]=std::int32_t(coordinate);
                if(auto next=layout_.index(neighbor))affected.insert(*next);
            }
        }
        if(staged.empty())return Edit::Unchanged;
        // Finish allocations before publication. As with scalar edits, a failed
        // metadata allocation may leave zero records, never partial occupancy.
        for(auto slot:affected)chunks_.try_emplace(slot);
        for(auto& [slot,prepared]:staged) {
            auto& current=chunks_.at(slot);
            occupied_-=current.occupied;occupied_+=prepared.occupied;
            if(current.voxels)--allocated_;
            if(!prepared.occupied)prepared.voxels.reset();else ++allocated_;
            current.voxels.swap(prepared.voxels);current.occupied=prepared.occupied;
            current.revisions.content=revision_+1;
        }
        ++revision_;
        for(auto slot:affected)chunks_.at(slot).revisions.surface=revision_;
        return Edit::Changed;
    }

private:
    using Voxels=std::array<std::uint32_t,voxelsPerChunk>;
    struct Entry {
        std::unique_ptr<Voxels> voxels;
        Revisions revisions;
        std::uint32_t occupied=0;
    };
    ResidentChunkLayout layout_;
    std::uint32_t materials_,allocated_=0;
    std::uint64_t revision_=0,occupied_=0;
    // Keep revision records after deletion so stale caches cannot become valid
    // again when a previously emptied chunk is recreated.
    std::map<std::uint32_t,Entry> chunks_;
};
}
