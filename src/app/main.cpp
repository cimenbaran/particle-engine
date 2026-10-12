#include "platform/Window.hpp"
#include "core/FixedStep.hpp"
#include "core/ParticleTypes.hpp"
#include "render/ImGuiRenderer.hpp"
#include "render/Shaders.hpp"
#include "render/GpuLayoutTest.hpp"
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <bx/math.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>

namespace {
constexpr uint32_t particleCount = 16384;
struct Graphics {
    Graphics(pw::Window& window) {
        int w, h; window.drawable(w,h);
        bgfx::Init init;
#if defined(__APPLE__)
        init.type = bgfx::RendererType::Metal;
#else
        init.type = bgfx::RendererType::Direct3D11;
#endif
        init.swapChain.nwh = window.nativeHandle();
        init.fallback = false;
        init.profile = true;
        init.swapChain.width = uint32_t(w); init.swapChain.height = uint32_t(h);
        init.reset = BGFX_RESET_VSYNC;
        if (!bgfx::init(init)) throw std::runtime_error("bgfx initialization failed for the required platform backend");
    }
    ~Graphics() { bgfx::shutdown(); }
};
struct GuiContext {
    explicit GuiContext(SDL_Window* window) {
        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        if (!ImGui_ImplSDL3_InitForOther(window)) { ImGui::DestroyContext(); throw std::runtime_error("SDL ImGui initialization failed"); }
    }
    ~GuiContext() { ImGui_ImplSDL3_Shutdown(); ImGui::DestroyContext(); }
};
struct Particles {
    bgfx::DynamicVertexBufferHandle state[2]{BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE};
    bgfx::DynamicVertexBufferHandle instances = BGFX_INVALID_HANDLE;
    bgfx::VertexBufferHandle vertices = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle compute = BGFX_INVALID_HANDLE, initialize = BGFX_INVALID_HANDLE, draw = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uniform = BGFX_INVALID_HANDLE;
    unsigned current{};
    explicit Particles(const std::filesystem::path& assets) {
        try {
            compute = pw::loadCompute(assets, "cs_particles");
            initialize = pw::loadCompute(assets, "cs_reset");
            draw = pw::loadProgram(assets, "vs_particles", "fs_particles");
            uniform = bgfx::createUniform("u_sim", bgfx::UniformType::Vec4);
            bgfx::VertexLayout layout;
            layout.begin().add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float).end();
            const float quad[]{-1,-1,0, 1,-1,0, 1,1,0, -1,1,0};
            const uint16_t triangles[]{0,1,2,0,2,3};
            vertices = bgfx::createVertexBuffer(bgfx::copy(quad, sizeof(quad)), layout);
            indices = bgfx::createIndexBuffer(bgfx::copy(triangles, sizeof(triangles)));
            if(!bgfx::isValid(vertices) || !bgfx::isValid(indices) || !bgfx::isValid(uniform))
                throw std::runtime_error("Particle rendering resources unavailable");
            reset();
        } catch (...) { release(); throw; }
    }
    void release() {
        for (auto h: state) if(bgfx::isValid(h)) bgfx::destroy(h);
        if(bgfx::isValid(instances)) bgfx::destroy(instances);
        if(bgfx::isValid(vertices)) bgfx::destroy(vertices);
        if(bgfx::isValid(indices)) bgfx::destroy(indices);
        if(bgfx::isValid(compute)) bgfx::destroy(compute);
        if(bgfx::isValid(initialize)) bgfx::destroy(initialize);
        if(bgfx::isValid(draw)) bgfx::destroy(draw);
        if(bgfx::isValid(uniform)) bgfx::destroy(uniform);
    }
    ~Particles() { release(); }
    void reset() {
        for (auto& h: state) { if(bgfx::isValid(h)) bgfx::destroy(h); h = BGFX_INVALID_HANDLE; }
        if(bgfx::isValid(instances)) bgfx::destroy(instances);
        instances = BGFX_INVALID_HANDLE;
        std::vector<pw::ParticleGpu> initial(particleCount);
        std::mt19937 rng(42);
        // Explicit integer mapping keeps authored initial positions reproducible.
        auto random = [&] { return float(rng() >> 8) / 16777216.0f; };
        for(uint32_t i=0; i<particleCount; ++i) {
            auto& p=initial[i];
            p.positionRadius[0]=(random()-0.5f)*12;
            p.positionRadius[1]=1+random()*10;
            p.positionRadius[2]=(random()-0.5f)*12;
            p.positionRadius[3]=0.045f;
            p.velocityInvMass[3]=1;
            p.thermalState[0]=293.15f;
            p.identity[0]=1; p.identity[1]=1; p.identity[2]=i; p.identity[3]=1;
        }
        bgfx::VertexLayout lanes;
        lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h: state)
            h=bgfx::createDynamicVertexBuffer(particleCount*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        bgfx::VertexLayout renderLayout;
        renderLayout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord1,4,bgfx::AttribType::Float).end();
        instances=bgfx::createDynamicVertexBuffer(particleCount,renderLayout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        for(auto h: state) if(!bgfx::isValid(h)) throw std::runtime_error("Particle state allocation failed");
        if(!bgfx::isValid(instances)) throw std::runtime_error("Instance allocation failed");
        auto upload=bgfx::createDynamicVertexBuffer(bgfx::copy(initial.data(), uint32_t(initial.size()*sizeof(pw::ParticleGpu))),lanes,BGFX_BUFFER_COMPUTE_READ);
        if(!bgfx::isValid(upload)) throw std::runtime_error("Seed upload allocation failed");
        const float params[]{0,float(particleCount),0,0};
        bgfx::setUniform(uniform,params);
        bgfx::setBuffer(0,upload,bgfx::Access::Read);
        bgfx::setBuffer(1,state[0],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::dispatch(0,initialize,(particleCount+127)/128);
        bgfx::destroy(upload); // bgfx defers destruction past submitted work.
        current=0;
    }
    void tick(uint16_t view) {
        bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
        const float params[]{float(pw::FixedStep::tickSeconds),float(particleCount),0,0};
        bgfx::setUniform(uniform, params);
        bgfx::setBuffer(0,state[current],bgfx::Access::Read);
        bgfx::setBuffer(1,state[1-current],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::dispatch(view,compute,(particleCount+127)/128);
        current=1-current;
    }
    void render(uint16_t view) {
        bgfx::setVertexBuffer(0,vertices); bgfx::setIndexBuffer(indices);
        bgfx::setInstanceDataBuffer(instances,0,particleCount);
        bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);
        bgfx::submit(view,draw);
    }
};
}
int main(int argc, char** argv) {
    try {
        uint32_t frameLimit=0; bool gpuTestOnly=false;
        for(int i=1;i<argc;++i) {
            if(std::string(argv[i])=="--frames" && i+1<argc) frameLimit=std::stoul(argv[++i]);
            else if(std::string(argv[i])=="--gpu-test") gpuTestOnly=true;
            else throw std::runtime_error("Usage: ParticleWorld [--frames count] [--gpu-test]");
        }
        pw::Window window;
        Graphics graphics(window);
        const auto* caps=bgfx::getCaps();
        const uint64_t required=BGFX_CAPS_COMPUTE;
        std::cout << "Backend: " << bgfx::getRendererName(caps->rendererType) << '\n';
        if((caps->supported&required)!=required || caps->limits.maxInstanceData < 2 || caps->limits.maxComputeBindings < 3) throw std::runtime_error("Required compute/instancing unavailable; cannot run GPU simulation");
        std::cout << "Compute and instancing supported; indirect draws "
            << ((caps->supported&BGFX_CAPS_DRAW_INDIRECT)?"available":"unavailable") << '\n';
        std::cout << "Submission: fixed initialized instance count; workgroup: 128\n";
        const char* basePath=SDL_GetBasePath();
        if(!basePath) throw std::runtime_error("SDL cannot resolve application assets");
        const std::filesystem::path assets=std::filesystem::path(basePath)/"assets";
        pw::runGpuLayoutTest(assets);
        if(gpuTestOnly) return 0;
        GuiContext gui(window.get());
        pw::ImGuiRenderer guiRenderer(assets);
        Particles particles(assets);
        pw::FixedStep clock;
        bool running=true,paused=false;
        uint64_t ticks=0; uint32_t frames=0; double lag=0;
        int width=0,height=0;
        float yaw=0.65f, distance=24, elevation=12;
        auto previous=std::chrono::steady_clock::now();
        while(running) {
            SDL_Event event;
            while(SDL_PollEvent(&event)) {
                // A button event carries its own position. Preserve it before queuing
                // the button, including synthetic clicks without a preceding motion event.
                if((event.type==SDL_EVENT_MOUSE_BUTTON_DOWN || event.type==SDL_EVENT_MOUSE_BUTTON_UP) &&
                    event.button.windowID==SDL_GetWindowID(window.get()))
                    ImGui::GetIO().AddMousePosEvent(event.button.x,event.button.y);
                ImGui_ImplSDL3_ProcessEvent(&event);
                if(std::getenv("PW_TRACE_INPUT") && (event.type==SDL_EVENT_MOUSE_BUTTON_DOWN || event.type==SDL_EVENT_MOUSE_BUTTON_UP))
                    std::cerr << "Mouse " << event.type << " at " << event.button.x << "," << event.button.y << " down=" << event.button.down << '\n';
                if(event.type==SDL_EVENT_QUIT || event.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED) running=false;
            }
            if(!running) break;
            auto now=std::chrono::steady_clock::now();
            double elapsed=std::chrono::duration<double>(now-previous).count(); previous=now;
            int w,h; window.drawable(w,h);
            if(w<=0 || h<=0 || (SDL_GetWindowFlags(window.get())&SDL_WINDOW_MINIMIZED)) {
                clock.reset(); SDL_Delay(16); continue;
            }
            if(w!=width || h!=height) { width=w;height=h; bgfx::SwapChain swap; swap.nwh=window.nativeHandle(); swap.width=uint32_t(w); swap.height=uint32_t(h); bgfx::reset(BGFX_RESET_VSYNC,&swap); }
            ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame();
            if(std::getenv("PW_TRACE_INPUT") && (ImGui::IsMouseClicked(0) || ImGui::IsMouseReleased(0)))
                std::cerr << "ImGui mouse " << ImGui::GetIO().MousePos.x << "," << ImGui::GetIO().MousePos.y << " clicked=" << ImGui::IsMouseClicked(0) << " released=" << ImGui::IsMouseReleased(0) << '\n';
            ImGui::SetNextWindowPos(ImVec2(16,16),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(350,320),ImGuiCond_FirstUseEver);
            ImGui::Begin("Particle World - foundation");
            ImGui::Text("Backend: %s",bgfx::getRendererName(caps->rendererType));
            ImGui::Text("Particles: %u | Seed: 42",particleCount);
            ImGui::Text("Tick: %llu",static_cast<unsigned long long>(ticks));
            ImGui::Checkbox("Paused",&paused); ImGui::SameLine();
            bool step=ImGui::Button("Single tick"); if(step) paused=true;
            if(ImGui::Button("Reset scene")) {particles.reset();clock.reset();ticks=0;lag=0;}
            ImGui::SliderFloat("Orbit",&yaw,-3.14f,3.14f);
            ImGui::SliderFloat("Distance",&distance,8,45);
            ImGui::SliderFloat("Elevation",&elevation,2,25);
            const auto* stats=bgfx::getStats();
            if(stats->cpuTimerFreq) ImGui::Text("CPU frame: %.2f ms",1000.0*double(stats->cpuTimeFrame)/double(stats->cpuTimerFreq));
            if(stats->gpuTimerFreq) ImGui::Text("GPU frame: %.2f ms",1000.0*double(stats->gpuTimeEnd-stats->gpuTimeBegin)/double(stats->gpuTimerFreq));
            ImGui::Text("Dropped wall time: %.3f s",lag);
            ImGui::TextWrapped("Foundation demo: gravity and box contacts. Pair collisions and GPU allocation are pending.");
            ImGui::End();
            auto batch=clock.advance(elapsed,paused,step); lag+=batch.droppedSeconds;
            for(unsigned i=0;i<batch.count;++i) {particles.tick(uint16_t(i+1));++ticks;}
            float view[16],projection[16];
            bx::mtxLookAt(view,{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance},{0,3,0},{0,1,0},bx::Handedness::Right);
            bx::mtxProj(projection,55,float(width)/float(height),0.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
            bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
            bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
            bgfx::setViewTransform(10,view,projection); bgfx::touch(10);
            particles.render(10);
            ImGui::Render(); guiRenderer.render(ImGui::GetDrawData(),20);
            bgfx::frame();
            if(frameLimit && ++frames>=frameLimit) running=false;
        }
        std::cout << "Clean shutdown after " << ticks << " ticks\n";
        return 0;
    } catch(const std::exception& error) {std::cerr << "Particle World: " << error.what() << '\n';return 1;}
}
