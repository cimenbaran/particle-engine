#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pw {
// Shader ABI: explicit lanes, independent of GLM build options.
struct alignas(16) ParticleGpu {
    float positionRadius[4];
    float velocityInvMass[4];
    float thermalState[4]; // kelvin, normalized fuel/water, seconds
    std::uint32_t identity[4]; // material, flags, stable ID, slot generation
};
struct alignas(16) RenderInstance {
    float positionRadius[4];
    std::uint32_t visual[4]; // packed color, class, stable ID, flags
};
struct ParticleId {
    std::uint32_t slot;
    std::uint32_t generation;
    bool operator==(const ParticleId&) const = default;
};
static_assert(std::is_standard_layout_v<ParticleGpu>);
static_assert(sizeof(ParticleGpu) == 64 && alignof(ParticleGpu) == 16);
static_assert(offsetof(ParticleGpu, positionRadius) == 0);
static_assert(offsetof(ParticleGpu, velocityInvMass) == 16);
static_assert(offsetof(ParticleGpu, thermalState) == 32);
static_assert(offsetof(ParticleGpu, identity) == 48);
static_assert(sizeof(RenderInstance) == 32 && alignof(RenderInstance) == 16);
}
