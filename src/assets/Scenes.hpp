#pragma once
#include "Materials.hpp"
#include "core/ParticleTypes.hpp"
#include <optional>
namespace pw {
struct ScenePreset {
    std::string name="Ballistic cloud";
    std::uint32_t seed=42;
    std::optional<std::uint32_t> count; // absent means fill the selected capacity
    std::array<float,3> minimum{-6,1,-6},maximum{6,11,6};
    std::vector<std::string> materials{"Stone","Sand","Wood","Ash"};
};
ScenePreset parseScene(std::string_view text);
std::string encodeScene(const ScenePreset& scene);
ScenePreset loadScene(const std::filesystem::path& path);
std::vector<std::uint32_t> sceneMaterials(const ScenePreset& scene,const MaterialLibrary& library);
void validateSceneCapacity(const ScenePreset& scene,std::uint32_t capacity);
std::vector<ParticleGpu> instantiateScene(const ScenePreset& scene,const MaterialLibrary& library,
    std::uint32_t capacity,std::span<const std::uint32_t> materials,std::optional<std::uint32_t> countOverride={});
}
