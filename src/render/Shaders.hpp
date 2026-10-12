#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
#include <string_view>
namespace pw {
bgfx::ProgramHandle loadProgram(const std::filesystem::path& assets, std::string_view vertex, std::string_view fragment);
bgfx::ProgramHandle loadCompute(const std::filesystem::path& assets, std::string_view name);
}
