#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
namespace pw {
enum class SpawnPlacement { Random, GranularLattice };
// Closed brush bounds; .125m global lattice gives nonoverlapping .0625m grains.
// Local coordinates are restricted to the voxel-query domain. No allocation.
struct SpawnLattice {
    std::array<std::int32_t,3> first{};
    std::array<std::uint32_t,3> dimensions{};
    std::uint32_t sites=0;
};
inline SpawnLattice spawnLattice(const std::array<float,3>& minimum,const std::array<float,3>& maximum) {
    SpawnLattice result;std::uint64_t sites=1;
    for(unsigned axis=0;axis<3;++axis) {
        if(!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]) || minimum[axis]>maximum[axis] || minimum[axis]<-64 || maximum[axis]>64)
            throw std::invalid_argument("Lattice brush requires finite ordered bounds in [-64,64]");
        // Double arithmetic preserves signed subnormal command boundaries on Metal.
        const auto first=std::int32_t(std::ceil(double(minimum[axis])*8));
        const auto last=std::int32_t(std::floor(double(maximum[axis])*8));
        result.first[axis]=first;result.dimensions[axis]=last>=first?std::uint32_t(last-first+1):0;
        sites*=result.dimensions[axis];
    }
    result.sites=std::uint32_t(sites);return result; // <=1025^3, fits uint32.
}
inline std::uint32_t spawnHash(std::uint32_t x) {
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;
}
inline std::optional<std::array<float,3>> spawnLatticePosition(const SpawnLattice& lattice,std::uint32_t request,std::uint32_t seed) {
    if(request>=lattice.sites)return {};
    std::uint64_t product=1;
    for(unsigned axis=0;axis<3;++axis) {
        if(!lattice.dimensions[axis] || lattice.dimensions[axis]>1025 || lattice.first[axis]<-512 || lattice.first[axis]>512 ||
           std::int64_t(lattice.first[axis])+lattice.dimensions[axis]-1>512)
            throw std::invalid_argument("Malformed spawn lattice");
        product*=lattice.dimensions[axis];
    }
    if(product!=lattice.sites)throw std::invalid_argument("Malformed spawn lattice size");
    auto index=(spawnHash(seed)%lattice.sites+request)%lattice.sites;
    std::array<float,3> result{};
    for(unsigned axis=0;axis<3;++axis){result[axis]=float(lattice.first[axis]+std::int32_t(index%lattice.dimensions[axis]))*.125f;index/=lattice.dimensions[axis];}
    return result;
}
}
