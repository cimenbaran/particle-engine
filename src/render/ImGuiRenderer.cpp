#include "ImGuiRenderer.hpp"
#include "Shaders.hpp"
#include <imgui.h>
#include <bx/math.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace pw {
ImGuiRenderer::ImGuiRenderer(const std::filesystem::path& assets) {
    layout_.begin().add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true).end();
    try {
        program_ = loadProgram(assets, "vs_gui", "fs_gui");
        sampler_ = bgfx::createUniform("s_texture", bgfx::UniformType::Sampler);
        unsigned char* pixels; int width, height;
        auto& io = ImGui::GetIO();
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        font_ = bgfx::createTexture2D(static_cast<uint16_t>(width), static_cast<uint16_t>(height),
            false, 1, bgfx::TextureFormat::RGBA8, 0, bgfx::copy(pixels, width*height*4));
        if (!bgfx::isValid(font_) || !bgfx::isValid(sampler_)) throw std::runtime_error("ImGui resources unavailable");
        io.Fonts->SetTexID(static_cast<ImTextureID>(font_.idx + 1));
        io.BackendRendererName = "particleworld_bgfx";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    } catch (...) { release(); throw; }
}
ImGuiRenderer::~ImGuiRenderer() {
    auto& io = ImGui::GetIO();
    io.Fonts->SetTexID(0); io.BackendRendererName = nullptr;
    io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;
    release();
}
void ImGuiRenderer::release() {
    if (bgfx::isValid(font_)) bgfx::destroy(font_);
    if (bgfx::isValid(sampler_)) bgfx::destroy(sampler_);
    if (bgfx::isValid(program_)) bgfx::destroy(program_);
}
void ImGuiRenderer::render(ImDrawData* data, std::uint16_t view) {
    const int width = int(data->DisplaySize.x * data->FramebufferScale.x);
    const int height = int(data->DisplaySize.y * data->FramebufferScale.y);
    if (width <= 0 || height <= 0) return;
    float ortho[16];
    bx::mtxOrtho(ortho, data->DisplayPos.x, data->DisplayPos.x + data->DisplaySize.x,
        data->DisplayPos.y + data->DisplaySize.y, data->DisplayPos.y, 0, 100, 0,
        bgfx::getCaps()->homogeneousDepth);
    bgfx::setViewMode(view, bgfx::ViewMode::Sequential);
    bgfx::setViewRect(view, 0, 0, uint16_t(width), uint16_t(height));
    bgfx::setViewTransform(view, nullptr, ortho);
    for (const ImDrawList* list : data->CmdLists) {
        const auto vertices = uint32_t(list->VtxBuffer.Size), indices = uint32_t(list->IdxBuffer.Size);
        if (bgfx::getAvailTransientVertexBuffer(vertices, layout_) != vertices ||
            bgfx::getAvailTransientIndexBuffer(indices, sizeof(ImDrawIdx)==4) != indices) break;
        bgfx::TransientVertexBuffer vb; bgfx::TransientIndexBuffer ib;
        bgfx::allocTransientVertexBuffer(&vb, vertices, layout_);
        bgfx::allocTransientIndexBuffer(&ib, indices, sizeof(ImDrawIdx)==4);
        std::memcpy(vb.data, list->VtxBuffer.Data, vertices * sizeof(ImDrawVert));
        std::memcpy(ib.data, list->IdxBuffer.Data, indices * sizeof(ImDrawIdx));
        for (const ImDrawCmd& command : list->CmdBuffer) {
            if (command.UserCallback) {
                if (command.UserCallback != ImDrawCallback_ResetRenderState) command.UserCallback(list, &command);
                continue;
            }
            const auto clip = command.ClipRect;
            const float x0 = std::clamp((clip.x-data->DisplayPos.x)*data->FramebufferScale.x, 0.f, float(width));
            const float y0 = std::clamp((clip.y-data->DisplayPos.y)*data->FramebufferScale.y, 0.f, float(height));
            const float x1 = std::clamp((clip.z-data->DisplayPos.x)*data->FramebufferScale.x, 0.f, float(width));
            const float y1 = std::clamp((clip.w-data->DisplayPos.y)*data->FramebufferScale.y, 0.f, float(height));
            if (x1 <= x0 || y1 <= y0 || command.GetTexID() == 0) continue;
            bgfx::setScissor(uint16_t(x0), uint16_t(y0), uint16_t(x1-x0), uint16_t(y1-y0));
            bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_MSAA |
                BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA));
            bgfx::setTexture(0, sampler_, bgfx::TextureHandle{uint16_t(command.GetTexID()-1)});
            bgfx::setVertexBuffer(0, &vb, command.VtxOffset, vertices-command.VtxOffset);
            bgfx::setIndexBuffer(&ib, command.IdxOffset, command.ElemCount);
            bgfx::submit(view, program_);
        }
    }
}
}
