#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
namespace pw {
enum class CommandType : std::uint32_t { SpawnBrush, ApplyImpulse, ApplyHeat, EditTerrain };
struct Command {
    std::uint64_t tick{}, sequence{};
    CommandType type{};
    std::array<float, 3> minimum{}, maximum{};
    std::array<float, 4> parameters{};
    std::uint32_t material{}, count{};
};
struct CommandLimits { std::uint32_t materialCount{7}, spawnPerCommand{4096}; float worldBound{1024}; };
inline std::string_view validate(const Command& c, const CommandLimits& limits,
                                 std::uint64_t currentTick) {
    if (c.tick < currentTick) return "command is in the past";
    if (static_cast<std::uint32_t>(c.type) > static_cast<std::uint32_t>(CommandType::EditTerrain))
        return "unknown command type";
    if (c.material >= limits.materialCount) return "invalid material ID";
    if (c.count > limits.spawnPerCommand) return "spawn quota exceeded";
    for (unsigned i = 0; i != 3; ++i) {
        if (!std::isfinite(c.minimum[i]) || !std::isfinite(c.maximum[i]) ||
            c.minimum[i] > c.maximum[i] || std::abs(c.minimum[i]) > limits.worldBound ||
            std::abs(c.maximum[i]) > limits.worldBound) return "invalid command bounds";
    }
    for (float p : c.parameters) if (!std::isfinite(p)) return "non-finite parameter";
    return {};
}
}
