#pragma once
#include "VoxelCoordinates.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <span>

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
