#pragma once
#include "Materials.hpp"
#include "core/TerrainExplosion.hpp"
namespace pw {
inline std::vector<terrain::ExplosionMaterial> explosionMaterials(const MaterialLibrary& library) {
    std::vector<terrain::ExplosionMaterial> result;
    result.reserve(library.entries.size());
    for(const auto& m:library.entries)
        result.push_back({m.gpu.mechanical[0],m.gpu.identity[1]==0,m.fractureResistance,m.gpu.thermal[3]>0?1.f:0.f});
    terrain::validateExplosionMaterials(result);
    return result;
}
}
