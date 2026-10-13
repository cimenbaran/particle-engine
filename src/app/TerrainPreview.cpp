#include "TerrainPreview.hpp"
#include "platform/Window.hpp"
#include "core/BasinWorld.hpp"
#include "core/FixedStep.hpp"
#include "render/VoxelProbes.hpp"
#include <chrono>
#include "render/VoxelTerrain.hpp"
#include "render/ImGuiRenderer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <bx/math.h>
#include <cmath>
#include <iostream>
namespace pw {
namespace {
struct PreviewGui {
    explicit PreviewGui(SDL_Window* window) {
        IMGUI_CHECKVERSION();ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;ImGui::StyleColorsDark();
        if(!ImGui_ImplSDL3_InitForOther(window)){ImGui::DestroyContext();throw std::runtime_error("Terrain GUI initialization failed");}
    }
    ~PreviewGui(){ImGui_ImplSDL3_Shutdown();ImGui::DestroyContext();}
};
struct BasinPublication {
    std::unique_ptr<terrain::SparseVoxelStore> store;
    std::unique_ptr<GpuVoxelWorld> occupancy;
    std::unique_ptr<VoxelTerrain> mesh;
    BasinPublication(const std::filesystem::path& assets,const MaterialLibrary& materials,std::uint32_t seed) {
        const auto id=[&](const char* name){for(unsigned i=0;i<materials.entries.size();++i)if(materials.entries[i].name==name)return i;throw std::runtime_error("Basin material missing");};
        store=terrain::buildBasin(seed,std::uint32_t(materials.entries.size()),id("Stone"),id("Sand"),id("Wood"));
        occupancy=std::make_unique<GpuVoxelWorld>(*store,terrain::Chunk{});
        mesh=std::make_unique<VoxelTerrain>(assets,*store,*occupancy,terrain::Chunk{},materials);
    }
};
}
int runTerrainPreview(Window& window,const std::filesystem::path& assets,std::uint32_t frameLimit) {
    SDL_SetWindowTitle(window.get(),"Particle World | Resident basin");
    auto materials=loadMaterials(assets/"materials/default.json");
    int seed=1337;auto basin=std::make_unique<BasinPublication>(assets,materials,std::uint32_t(seed));
    PreviewGui gui(window.get());ImGuiRenderer renderer(assets);
    float yaw=.65f,distance=43,elevation=30;bool running=true;std::uint32_t frames=0;
    int width=0,height=0;std::string message;
    std::unique_ptr<VoxelProbes> probes;FixedStep clock;auto previous=std::chrono::steady_clock::now();std::uint32_t completed=0;bool paused=false;
    std::cout<<"Resident basin: "<<basin->store->occupiedVoxelCount()<<" voxels, "<<basin->mesh->faceCount()<<" exposed faces, seed "<<seed<<"; terrain inspection with optional collision probes\n";
    while(running) {
        if(probes)probes->poll(completed);
        auto now=std::chrono::steady_clock::now();double elapsed=std::chrono::duration<double>(now-previous).count();previous=now;
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            if((event.type==SDL_EVENT_MOUSE_BUTTON_DOWN || event.type==SDL_EVENT_MOUSE_BUTTON_UP) && event.button.windowID==SDL_GetWindowID(window.get()))
                ImGui::GetIO().AddMousePosEvent(event.button.x,event.button.y);
            ImGui_ImplSDL3_ProcessEvent(&event);
            if(event.type==SDL_EVENT_QUIT || event.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED ||
               (event.type==SDL_EVENT_KEY_DOWN && event.key.scancode==SDL_SCANCODE_ESCAPE))running=false;
        }
        if(!running)break;
        int w,h;window.drawable(w,h);
        if(w<=0 || h<=0 || (SDL_GetWindowFlags(window.get())&SDL_WINDOW_MINIMIZED)){SDL_Delay(16);continue;}
        if(w!=width || h!=height) {
            width=w;height=h;bgfx::SwapChain swap;swap.nwh=window.nativeHandle();swap.width=std::uint32_t(w);swap.height=std::uint32_t(h);bgfx::reset(BGFX_RESET_VSYNC,&swap);
        }
        ImGui_ImplSDL3_NewFrame();ImGui::NewFrame();
        ImGui::SetNextWindowPos({16,16},ImGuiCond_FirstUseEver);ImGui::SetNextWindowSize({360,470},ImGuiCond_FirstUseEver);
        ImGui::Begin("Resident basin");
        ImGui::TextUnformatted("32 x 16 x 32 metres | 0.125 m voxels");
        ImGui::TextWrapped("Terrain inspection with optional collision probes. Drop interacting sand to enable grain-to-grain contacts. Water and burning are not enabled here.");
        ImGui::TextWrapped("Stone ledge, sand slope, dry pool recess and wood fuel patch.");
        ImGui::Separator();
        ImGui::SliderAngle("Orbit",&yaw,-180,180);ImGui::SliderFloat("Distance",&distance,18,65,"%.1f m");ImGui::SliderFloat("Height",&elevation,5,50,"%.1f m");
        ImGui::InputInt("Seed",&seed);seed=std::max(0,seed);
        if(ImGui::Button("Rebuild basin")) {
            try {auto candidate=std::make_unique<BasinPublication>(assets,materials,std::uint32_t(seed));probes.reset();basin=std::move(candidate);clock.reset();message="Basin rebuilt";}
            catch(const std::exception& error){message=error.what();}
        }
        if(ImGui::Button("Drop interacting sand")){try{auto candidate=std::make_unique<VoxelProbes>(assets,materials,true);probes=std::move(candidate);clock.reset();paused=false;message="256 interacting grains; terrain and particle contacts";}catch(const std::exception& error){message=error.what();}}
        if(ImGui::Button("Drop collision probes")){try{auto candidate=std::make_unique<VoxelProbes>(assets,materials);probes=std::move(candidate);clock.reset();paused=false;message="256 terrain probes; no grain-to-grain contacts";}catch(const std::exception& error){message=error.what();}}
        if(probes){ImGui::Checkbox("Pause simulation",&paused);ImGui::Text("Simulation tick: %llu",static_cast<unsigned long long>(probes->ticks()));if(probes->fault())ImGui::TextWrapped("Physics fault: the tick was rolled back. Drop particles to reset.");}
        ImGui::Text("%llu occupied voxels | %u faces",static_cast<unsigned long long>(basin->store->occupiedVoxelCount()),basin->mesh->faceCount());
        ImGui::Text("Terrain GPU payload: %.1f MiB",double(basin->occupancy->payloadBytes()+basin->mesh->payloadBytes())/1048576);
        if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
        ImGui::End();
        const auto batch=clock.advance(elapsed,!probes || paused || probes->fault(),false);
        if(probes){for(unsigned tick=0;tick<batch.count;++tick)probes->tick(std::uint16_t(tick+1),*basin->occupancy,*basin->store);probes->prepare(5);if(probes->ticks()%8==0)probes->request(6);}
        float view[16],projection[16];
        bx::mtxLookAt(view,{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance},{0,1,0});
        bx::mtxProj(projection,50.0f,float(width)/float(height),.1f,200.0f,bgfx::getCaps()->homogeneousDepth);
        bgfx::setViewRect(10,0,0,std::uint16_t(width),std::uint16_t(height));
        bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x182333ff,1.0f,0);bgfx::setViewTransform(10,view,projection);bgfx::touch(10);
        basin->mesh->render(10,*basin->store,*basin->occupancy,{});
        if(probes)probes->render(10);
        ImGui::Render();renderer.render(ImGui::GetDrawData(),20);completed=bgfx::frame();
        if(frameLimit && ++frames>=frameLimit)break;
    }
    return 0;
}
}
