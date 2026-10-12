#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
struct ImDrawData;
namespace pw {
class ShaderReload;
void runImGuiRendererTest(const std::filesystem::path& assets);
class ImGuiRenderer {
public:
    explicit ImGuiRenderer(const std::filesystem::path& assets);
    ~ImGuiRenderer();
    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;
    void render(ImDrawData* data, std::uint16_t view);
    void stageShaders(ShaderReload& reload);
private:
    void release();
    bgfx::ProgramHandle program_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle sampler_ = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle font_ = BGFX_INVALID_HANDLE;
    bgfx::VertexLayout layout_;
};
}
