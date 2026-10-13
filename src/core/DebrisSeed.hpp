#pragma once
#include <array>
#include <cstdint>
namespace pw {
struct DebrisSeed {
    std::array<float,3> position{},velocity{};
    std::uint32_t material=0;
    float temperature=293.15f,fuel=0;
};
}
