#pragma once
#include "core/SpatialGrid.hpp"
#include <algorithm>
#include <span>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace pw::reference {
struct SpatialPoint {
    std::uint32_t slot;
    SpatialGrid::Position position;
    bool active=true,physical=true;
    std::uint32_t interactionClass=0; // Same-class contacts in this fixture profile.
    bool operator==(const SpatialPoint&) const=default;
};
struct CellSlot {
    std::uint32_t key,slot;
    bool operator==(const CellSlot&) const=default;
};
struct CellRange {
    std::uint32_t key,begin,end; // Half-open interval in sorted records.
    bool operator==(const CellRange&) const=default;
};
struct SpatialIndex {
    std::vector<CellSlot> records;
    std::vector<CellRange> ranges;
    std::span<const CellSlot> find(std::uint32_t key) const {
        const auto found=std::lower_bound(ranges.begin(),ranges.end(),key,
            [](const auto& range,auto value) {return range.key<value;});
        if(found==ranges.end() || found->key!=key) return {};
        return std::span<const CellSlot>(records).subspan(found->begin,found->end-found->begin);
    }
};
// Test oracle only: no physical state mutation and no hash collision ambiguity.
// Failure throws before publishing an index, never a partial valid-looking one.
inline SpatialIndex buildSpatialIndex(const SpatialGrid& grid,std::span<const SpatialPoint> points) {
    if(points.size()>UINT32_MAX) throw std::invalid_argument("Reference point count exceeds uint32");
    SpatialIndex index;
    std::unordered_set<std::uint32_t> slots;
    for(const auto& point:points) {
        if(point.slot==UINT32_MAX || !slots.insert(point.slot).second)
            throw std::invalid_argument("Invalid or duplicate reference slot");
        if(!point.active || !point.physical) continue;
        const auto key=grid.keyAt(point.position);
        if(!key) throw std::invalid_argument("Active physical point has no valid grid cell");
        index.records.push_back({*key,point.slot});
    }
    // Within-cell slot order is canonical for comparison; GPU radix sorting is
    // only required to preserve membership, not this tie order.
    std::sort(index.records.begin(),index.records.end(),[](const auto& a,const auto& b) {
        return a.key<b.key || (a.key==b.key && a.slot<b.slot);
    });
    for(std::size_t begin=0;begin<index.records.size();) {
        std::size_t end=begin+1;
        while(end<index.records.size() && index.records[end].key==index.records[begin].key) ++end;
        index.ranges.push_back({index.records[begin].key,std::uint32_t(begin),std::uint32_t(end)});
        begin=end;
    }
    return index;
}
// Owns a snapshot: querying an index with positions from another solver
// iteration is intentionally impossible through this reference interface.
class NeighborReference {
public:
    NeighborReference(SpatialGrid grid,std::span<const SpatialPoint> points)
        :grid_(grid),points_(points.begin(),points.end()),index_(buildSpatialIndex(grid_,points_)) {
        for(std::size_t i=0;i<points_.size();++i) bySlot_.emplace(points_[i].slot,i);
    }
    const SpatialIndex& index() const {return index_;}
    std::vector<std::uint32_t> neighbors(std::uint32_t slot,float support) const {
        const auto radius=grid_.searchRadius(support);
        const auto found=bySlot_.find(slot);
        if(found==bySlot_.end()) throw std::invalid_argument("Unknown query slot");
        const auto& query=points_[found->second];
        if(!query.active || !query.physical) return {};
        const auto cell=*grid_.coordinates(query.position);
        const double supportSquared=double(support)*support;
        std::vector<std::uint32_t> result;
        // Test fixtures scan occupied cells rather than allocating dense tables
        // or enumerating potentially huge empty search boxes. The same integer
        // support bounds apply to a future GPU neighbor-cell enumeration.
        for(const auto& range:index_.ranges) {
            const auto candidateCell=*grid_.decode(range.key);
            bool inRange=true;
            for(unsigned axis=0;axis<3;++axis)
                if(std::abs(std::int64_t(candidateCell[axis])-cell[axis])>radius) inRange=false;
            if(!inRange) continue;
            for(std::uint32_t i=range.begin;i<range.end;++i) {
                const auto& candidate=points_[bySlot_.at(index_.records[i].slot)];
                if(candidate.slot==slot || candidate.interactionClass!=query.interactionClass) continue;
                double distanceSquared=0;
                for(unsigned axis=0;axis<3;++axis) {
                    const double delta=double(candidate.position[axis])-query.position[axis];
                    distanceSquared+=delta*delta;
                }
                if(distanceSquared<=supportSquared) result.push_back(candidate.slot);
            }
        }
        std::sort(result.begin(),result.end());
        return result;
    }
private:
    SpatialGrid grid_;
    std::vector<SpatialPoint> points_;
    SpatialIndex index_;
    std::unordered_map<std::uint32_t,std::size_t> bySlot_;
};
}
