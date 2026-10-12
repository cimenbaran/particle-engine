#pragma once
#include "core/SpatialGrid.hpp"
#include <algorithm>
#include <span>
#include <unordered_set>
#include <vector>

namespace pw::reference {
struct SpatialPoint {
    std::uint32_t slot;
    SpatialGrid::Position position;
    bool active=true,physical=true;
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
}
