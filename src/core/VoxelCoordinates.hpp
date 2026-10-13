#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace pw::terrain {
inline constexpr std::uint32_t chunkSide=32,voxelsPerChunk=32*32*32;
inline constexpr double voxelMetres=0.125,chunkMetres=4;
using Chunk=std::array<std::int32_t,3>;
using LocalVoxel=std::array<std::uint8_t,3>;
using GlobalVoxel=std::array<std::int64_t,3>;
struct VoxelAddress {
    Chunk chunk{};
    LocalVoxel local{};
    bool operator==(const VoxelAddress&) const=default;
};
inline std::optional<std::uint32_t> localIndex(LocalVoxel local) {
    for(auto value:local)if(value>=chunkSide)return {};
    return local[0]+chunkSide*(local[1]+chunkSide*local[2]);
}
inline std::optional<LocalVoxel> decodeLocal(std::uint32_t index) {
    if(index>=voxelsPerChunk)return {};
    return LocalVoxel{std::uint8_t(index%32),std::uint8_t((index/32)%32),std::uint8_t(index/1024)};
}
// Euclidean division, including negative coordinates. No signed shifts or
// negation of INT64_MIN; out-of-range chunk coordinates fail before narrowing.
inline std::optional<VoxelAddress> addressOf(GlobalVoxel voxel) {
    VoxelAddress result;
    for(unsigned axis=0;axis<3;++axis) {
        auto chunk=voxel[axis]/32,remainder=voxel[axis]%32;
        if(remainder<0){--chunk;remainder+=32;}
        if(chunk<INT32_MIN || chunk>INT32_MAX)return {};
        result.chunk[axis]=std::int32_t(chunk);result.local[axis]=std::uint8_t(remainder);
    }
    return result;
}
inline std::optional<GlobalVoxel> voxelOf(VoxelAddress address) {
    if(!localIndex(address.local))return {};
    GlobalVoxel result;
    for(unsigned axis=0;axis<3;++axis)result[axis]=std::int64_t(address.chunk[axis])*32+address.local[axis];
    return result;
}
// CPU coordinates relative to a shared chunk origin. Keep address arithmetic
// integer; never construct a large absolute float position and subtract it.
inline std::optional<std::array<double,3>> relativeCorner(VoxelAddress address,Chunk origin) {
    if(!localIndex(address.local))return {};
    std::array<double,3> result;
    for(unsigned axis=0;axis<3;++axis) {
        const auto relative=(std::int64_t(address.chunk[axis])-origin[axis])*32+address.local[axis];
        result[axis]=double(relative)*voxelMetres;
    }
    return result;
}
inline std::optional<VoxelAddress> addressAtRelative(std::array<double,3> metres,Chunk origin={}) {
    GlobalVoxel voxel;
    constexpr auto low=std::int64_t(INT32_MIN)*32,high=std::int64_t(INT32_MAX)*32+31;
    for(unsigned axis=0;axis<3;++axis) {
        const double scaled=metres[axis]*8;
        if(!std::isfinite(scaled))return {};
        const double global=std::floor(scaled)+double(std::int64_t(origin[axis])*32);
        if(global<double(low) || global>double(high))return {};
        voxel[axis]=std::int64_t(global);
    }
    return addressOf(voxel);
}
// Finite resident region, independent of visibility. Slot order is x-fastest.
// This is addressing only; neither residency policy nor an allocation occurs here.
class ResidentChunkLayout {
public:
    ResidentChunkLayout(Chunk minimum,std::array<std::uint32_t,3> dimensions)
        :minimum_(minimum),dimensions_(dimensions) {
        std::uint64_t count=1;
        for(unsigned axis=0;axis<3;++axis) {
            if(!dimensions[axis] || count>UINT32_MAX/dimensions[axis] ||
               std::int64_t(minimum[axis])+dimensions[axis]-1>INT32_MAX)
                throw std::invalid_argument("Resident chunk layout exceeds coordinate/index range");
            count*=dimensions[axis];
        }
        count_=std::uint32_t(count);
    }
    std::optional<std::uint32_t> index(Chunk chunk) const {
        std::array<std::uint64_t,3> offset;
        for(unsigned axis=0;axis<3;++axis) {
            const auto delta=std::int64_t(chunk[axis])-minimum_[axis];
            if(delta<0 || std::uint64_t(delta)>=dimensions_[axis])return {};
            offset[axis]=std::uint64_t(delta);
        }
        return std::uint32_t(offset[0]+dimensions_[0]*(offset[1]+dimensions_[1]*offset[2]));
    }
    std::optional<Chunk> decode(std::uint32_t index) const {
        if(index>=count_)return {};
        Chunk result;
        for(unsigned axis=0;axis<3;++axis){result[axis]=std::int32_t(std::int64_t(minimum_[axis])+index%dimensions_[axis]);index/=dimensions_[axis];}
        return result;
    }
    Chunk minimum() const{return minimum_;}
    std::array<std::uint32_t,3> dimensions() const{return dimensions_;}
    std::uint32_t chunkCount() const{return count_;}
    std::uint64_t voxelCount() const{return std::uint64_t(count_)*voxelsPerChunk;}
private:
    Chunk minimum_;
    std::array<std::uint32_t,3> dimensions_;
    std::uint32_t count_;
};
}
