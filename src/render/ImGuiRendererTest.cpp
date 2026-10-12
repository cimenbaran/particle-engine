#include "ImGuiRenderer.hpp"
#include <imgui.h>
#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace pw {
void runImGuiRendererTest(const std::filesystem::path& assets) {
    struct Context {
        ImGuiContext* previous=ImGui::GetCurrentContext();
        ImGuiContext* current=ImGui::CreateContext();
        Context() {ImGui::SetCurrentContext(current);}
        ~Context() {ImGui::DestroyContext(current);ImGui::SetCurrentContext(previous);}
    } context;
    ImGuiRenderer renderer(assets);
    struct Targets {
        bgfx::TextureHandle red=BGFX_INVALID_HANDLE,green=BGFX_INVALID_HANDLE;
        bgfx::TextureHandle color=BGFX_INVALID_HANDLE,readback=BGFX_INVALID_HANDLE;
        bgfx::FrameBufferHandle frame=BGFX_INVALID_HANDLE;
        ~Targets() {
            bgfx::setViewFrameBuffer(20,BGFX_INVALID_HANDLE);
            bgfx::setViewClear(20,BGFX_CLEAR_NONE);
            if(bgfx::isValid(frame)) bgfx::destroy(frame);
            for(auto texture:{red,green,color,readback}) if(bgfx::isValid(texture)) bgfx::destroy(texture);
        }
    } target;
    const std::array<std::uint8_t,4> red{255,0,0,255},green{0,255,0,128};
    target.red=bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA8,0,bgfx::copy(red.data(),4));
    target.green=bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA8,0,bgfx::copy(green.data(),4));
    target.color=bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_RT);
    target.readback=bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    for(auto texture:{target.red,target.green,target.color,target.readback})
        if(!bgfx::isValid(texture)) throw std::runtime_error("GUI image test texture unavailable");
    target.frame=bgfx::createFrameBuffer(1,&target.color,false);
    if(!bgfx::isValid(target.frame)) throw std::runtime_error("GUI image test framebuffer unavailable");

    // A nonzero viewport origin, 2x scale, padded vertices and padded indices
    // ensure that ignoring any projection/clip/offset input changes the image.
    ImDrawList list(ImGui::GetDrawListSharedData());
    for(int i=0;i<4;++i) list.VtxBuffer.push_back({{-1000,-1000},{0,0},IM_COL32_WHITE});
    for(auto position:{ImVec2(10,20),ImVec2(42,20),ImVec2(42,52),ImVec2(10,52)})
        list.VtxBuffer.push_back({position,{0.5f,0.5f},IM_COL32_WHITE});
    for(auto index:{0,0,0,0,0,0,0,1,2,0,2,3}) list.IdxBuffer.push_back(ImDrawIdx(index));
    ImDrawCmd command;command.ElemCount=6;command.VtxOffset=4;command.IdxOffset=6;
    command.ClipRect={10,20,26,36};command.TexRef=ImTextureRef(ImTextureID(target.red.idx+1));
    list.CmdBuffer.push_back(command);
    command.ClipRect={26,36,42,52};command.TexRef=ImTextureRef(ImTextureID(target.green.idx+1));
    list.CmdBuffer.push_back(command);
    ImDrawData data;data.Valid=true;data.DisplayPos={10,20};data.DisplaySize={32,32};data.FramebufferScale={2,2};
    data.CmdLists.push_back(&list);data.CmdListsCount=1;data.TotalVtxCount=8;data.TotalIdxCount=12;
    bgfx::setViewFrameBuffer(20,target.frame);
    bgfx::setViewClear(20,BGFX_CLEAR_COLOR,0x000000ff);
    bgfx::touch(20);renderer.render(&data,20);
    bgfx::TextureRegion src;src.handle=target.color;
    bgfx::TextureRegion dst;dst.handle=target.readback;
    std::array<std::uint8_t,64*64*4> pixels{};
    bgfx::blit(21,dst,src);const auto ready=bgfx::read(dst,pixels.data());
    while(bgfx::frame()<ready) SDL_PumpEvents();
    for(int y=0;y<64;++y) for(int x=0;x<64;++x) {
        // Ignore only rasterization boundaries, checking the rest of all four
        // quadrants rather than accepting a merely nonuniform image.
        if(x<2 || y<2 || x>61 || y>61 || (x>=30 && x<=33) || (y>=30 && y<=33)) continue;
        const int row=bgfx::getCaps()->originBottomLeft?63-y:y;
        const int offset=(row*64+x)*4;
        const int expectedRed=x<32 && y<32?255:0;
        const int expectedGreen=x>=32 && y>=32?128:0;
        if(std::abs(int(pixels[offset])-expectedRed)>2 ||
           std::abs(int(pixels[offset+1])-expectedGreen)>2 || pixels[offset+2]>2)
            throw std::runtime_error("GUI texture, alpha, clipping, offset or framebuffer-scale image mismatch at "+
                std::to_string(x)+","+std::to_string(y));
    }
    std::cout<<"GPU GUI custom textures, alpha blend, scissor, index/vertex offsets and 2x framebuffer scale passed\n";
}
}
