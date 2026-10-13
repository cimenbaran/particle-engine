#pragma once
#include <filesystem>
#include <cstdint>
namespace pw {
class Window;
int runTerrainPreview(Window& window,const std::filesystem::path& assets,std::uint32_t frameLimit);
}
