#pragma once
#include "SparseVoxelStore.hpp"
#include <vector>

namespace pw::terrain {
// Sides are -X,+X,-Y,+Y,-Z,+Z. Local indices are x-fast voxel indices.
struct VoxelFace {
    std::uint16_t voxel=0;
    std::uint8_t side=0;
    std::uint32_t material=0;
    bool operator==(const VoxelFace&) const=default;
};
struct VoxelSurface {
    const SparseVoxelStore* owner=nullptr;
    Chunk chunk{};
    SparseVoxelStore::Revisions revisions{};
    std::vector<VoxelFace> faces;
    // The owner must outlive this surface. This is CPU cache validity only.
    bool isCurrent(const SparseVoxelStore& store) const {
        return owner==&store && store.revisions(chunk)==revisions;
    }
};
using FaceCorners=std::array<std::array<std::uint8_t,3>,4>;
// Chunk-local integer grid corners, CCW viewed from outside. Triangles may use
// indices 0,1,2 and 0,2,3. Convert to metres only relative to a shared origin.
inline std::optional<FaceCorners> faceCorners(VoxelFace face) {
    auto local=decodeLocal(face.voxel);if(!local || face.side>=6)return {};
    const unsigned axis=face.side/2,u=(axis+1)%3,v=(axis+2)%3;
    FaceCorners corners;for(auto& corner:corners)corner=*local;
    if(face.side&1)for(auto& corner:corners)++corner[axis];
    if(face.side&1){++corners[1][u];++corners[3][v];}
    else {++corners[1][v];++corners[3][u];}
    ++corners[2][u];++corners[2][v];return corners;
}
// Every occupied voxel is opaque solid for this geometric extraction. Faces
// between occupied materials are hidden; outside the resident region is empty.
// The store's single-threaded ownership rule applies throughout this call.
inline std::optional<VoxelSurface> buildVoxelSurface(const SparseVoxelStore& store,Chunk chunk) {
    std::array<std::uint32_t,voxelsPerChunk> voxels;
    auto revisions=store.copyChunk(chunk,voxels);if(!revisions)return {};
    VoxelSurface result{&store,chunk,*revisions,{}};
    constexpr std::array<std::uint32_t,3> strides{1,32,1024};
    for(std::uint32_t index=0;index<voxelsPerChunk;++index) {
        if(voxels[index]==SparseVoxelStore::empty)continue;
        const auto local=*decodeLocal(index);
        for(unsigned side=0;side<6;++side) {
            const unsigned axis=side/2;const int step=(side&1)?1:-1;
            std::uint32_t neighbor=SparseVoxelStore::empty;
            if((step<0 && local[axis]>0) || (step>0 && local[axis]<31))
                neighbor=voxels[step<0?index-strides[axis]:index+strides[axis]];
            else {
                const auto coordinate=std::int64_t(chunk[axis])+step;
                if(coordinate>=INT32_MIN && coordinate<=INT32_MAX) {
                    VoxelAddress other{chunk,local};other.chunk[axis]=std::int32_t(coordinate);
                    other.local[axis]=step<0?31:0;
                    neighbor=store.materialAt(other).value_or(SparseVoxelStore::empty);
                }
            }
            if(neighbor==SparseVoxelStore::empty)
                result.faces.push_back({std::uint16_t(index),std::uint8_t(side),voxels[index]});
        }
    }
    return result;
}
}
