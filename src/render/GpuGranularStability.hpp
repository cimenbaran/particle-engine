#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>
namespace pw {
nlohmann::json runGpuGranularStability(const std::filesystem::path& assets);
}
