#include "Shaders.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>
namespace pw {
static bgfx::ShaderHandle shader(const std::filesystem::path& assets, std::string_view name) {
    auto path = assets / "shaders" / (std::string(name) + ".bin");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader: " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || size > 16 * 1024 * 1024) throw std::runtime_error("Invalid shader size");
    std::vector<char> data(static_cast<std::size_t>(size));
    file.seekg(0); file.read(data.data(), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Cannot read shader: " + path.string());
    auto handle = bgfx::createShader(bgfx::copy(data.data(), static_cast<std::uint32_t>(data.size())));
    if (!bgfx::isValid(handle)) throw std::runtime_error("Shader rejected: " + path.string());
    return handle;
}
bgfx::ProgramHandle loadCompute(const std::filesystem::path& assets, std::string_view name) {
    auto program = bgfx::createProgram(shader(assets, name), true);
    if (!bgfx::isValid(program)) throw std::runtime_error("Compute program creation failed");
    return program;
}
bgfx::ProgramHandle loadProgram(const std::filesystem::path& assets, std::string_view vertex, std::string_view fragment) {
    auto vs = shader(assets, vertex);
    bgfx::ShaderHandle fs = BGFX_INVALID_HANDLE;
    try { fs = shader(assets, fragment); } catch (...) { bgfx::destroy(vs); throw; }
    auto program = bgfx::createProgram(vs, fs, true);
    if (!bgfx::isValid(program)) throw std::runtime_error("Graphics program creation failed");
    return program;
}
}
