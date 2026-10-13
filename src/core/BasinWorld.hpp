#pragma once
#include "SparseVoxelStore.hpp"
namespace pw::terrain {
// Version 1 static basin recipe. Water and thermal state are deliberately absent
// until their simulation exists; the pool is an empty recess in the stone floor.
inline std::unique_ptr<SparseVoxelStore> buildBasin(std::uint32_t seed,std::uint32_t materialCount,
                                                  std::uint32_t stone,std::uint32_t sand,std::uint32_t fuel) {
    if(stone>=materialCount || sand>=materialCount || fuel>=materialCount || stone==sand || stone==fuel || sand==fuel)
        throw std::invalid_argument("Invalid basin materials");
    auto store=std::make_unique<SparseVoxelStore>(ResidentChunkLayout({-4,0,-4},{8,4,8}),materialCount);
    std::vector<SparseVoxelStore::VoxelEdit> edits;edits.reserve(400000);
    for(int z=-128;z<128;++z)for(int x=-128;x<128;++x) {
        const bool rim=x< -126 || x>=126 || z< -126 || z>=126;
        const bool pool=x>=24 && x<88 && z>=24 && z<88;
        const bool ledge=x>=-96 && x< -48 && z>=-64 && z<0;
        const bool slope=x>=-48 && x<0 && z>=-64 && z<0;
        const bool patch=x>=32 && x<48 && z>=-48 && z< -32;
        // Integer hash only: the topmost sand grain varies reproducibly by seed.
        std::uint32_t hash=seed^std::uint32_t(x)*0x9e3779b9u^std::uint32_t(z)*0x85ebca6bu;
        hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;
        const int base=pool?1:4;
        const int top=rim?16:ledge?32:slope?4+(-x*28)/48+int(hash&1u):patch?5:base;
        for(int y=0;y<top;++y) {
            const auto material=y<base || rim || ledge?stone:slope?sand:fuel;
            edits.push_back({*addressOf({x,y,z}),material});
        }
    }
    store->setMaterials(edits,0);return store;
}
}
