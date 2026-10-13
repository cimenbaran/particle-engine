#pragma once
#include <stdexcept>
#include <string>
#include <string_view>
namespace pw {
// Both profiles use 1/8 m cells and y=[0,16). Legacy x/z=[-8,8),
// Basin x/z=[-16,16). Raw index buffers must share the consumer's profile.
enum class GranularGridProfile { Legacy, Basin };
inline std::string gridShader(std::string_view name,GranularGridProfile profile) {
    switch(profile) {
    case GranularGridProfile::Legacy:return std::string(name);
    case GranularGridProfile::Basin:return std::string(name)+"_basin";
    }
    throw std::invalid_argument("Invalid granular grid profile");
}
}
