#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pw {
// Four 16-byte lanes; matches raw vec4 shader addressing. All temperatures are K.
struct alignas(16) MaterialGpu {
    float mechanical[4]; // density kg/m3, friction, restitution, drag 1/s
    float thermal[4]; // conductivity W/(m K), heat capacity J/(kg K), ignition K, burn rate 1/s
    float color[4]; // linear RGBA, normalized
    std::uint32_t identity[4]; // stable material ID, phase, render class, reserved
};
static_assert(sizeof(MaterialGpu)==64 && alignof(MaterialGpu)==16);
static_assert(offsetof(MaterialGpu,mechanical)==0 && offsetof(MaterialGpu,thermal)==16);
static_assert(offsetof(MaterialGpu,color)==32 && offsetof(MaterialGpu,identity)==48);
inline constexpr std::string_view materialSchema =
    "PWMaterial/1/le/64:mechanical=f32x4:thermal=f32x4:color=f32x4:identity=u32x4;units=m,kg,s,K;transitions=acyclic";
struct MaterialDesc {
    std::string name;
    MaterialGpu gpu{};
    std::vector<std::uint32_t> transitions;
};
struct MaterialLibrary {
    std::vector<MaterialDesc> entries;
    std::uint64_t contentHash{};
    std::vector<MaterialGpu> packed() const;
};
std::uint64_t fingerprint(std::string_view text);
MaterialLibrary parseMaterials(std::string_view text);
MaterialLibrary loadMaterials(const std::filesystem::path& path);
void validateMaterialReload(const MaterialLibrary& current, const MaterialLibrary& candidate,
                            std::span<const std::uint32_t> liveIds);
// Portable little-endian output, never native struct/padding serialization.
std::vector<std::byte> encodeMaterials(const MaterialLibrary& library);
}
