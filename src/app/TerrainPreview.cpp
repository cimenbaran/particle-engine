#include "TerrainPreview.hpp"
#include "platform/Window.hpp"
#include "core/BasinWorld.hpp"
#include "core/FixedStep.hpp"
#include "core/TerrainExcavation.hpp"
#include "render/PendingDebrisTransaction.hpp"
#include "render/StagedExplosion.hpp"
#include "render/VoxelProbes.hpp"
#include <chrono>
#include "render/VoxelPublication.hpp"
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
        IMGUI_CHECKVERSION();ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;ImGui::GetIO().ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;ImGui::StyleColorsDark();
        if(!ImGui_ImplSDL3_InitForOther(window)){ImGui::DestroyContext();throw std::runtime_error("Terrain GUI initialization failed");}
    }
    ~PreviewGui(){ImGui_ImplSDL3_Shutdown();ImGui::DestroyContext();}
};
struct BasinPublication {
    VoxelPublicationPtr publication;
    const terrain::SparseVoxelStore* store;
    const GpuVoxelWorld* occupancy;
    const VoxelTerrain* mesh;
    BasinPublication(const std::filesystem::path& assets,const MaterialLibrary& materials,std::uint32_t seed) {
        const auto id=[&](const char* name){for(unsigned i=0;i<materials.entries.size();++i)if(materials.entries[i].name==name)return i;throw std::runtime_error("Basin material missing");};
        auto source=terrain::buildBasin(seed,std::uint32_t(materials.entries.size()),id("Stone"),id("Sand"),id("Wood"));
        publication=std::make_shared<VoxelPublication>(assets,std::move(source),terrain::Chunk{},materials);
        refresh();
    }
    void refresh(){store=&publication->source();occupancy=&publication->occupancy();mesh=&publication->mesh();}
};
}
int runTerrainPreview(Window& window,const std::filesystem::path& assets,std::uint32_t frameLimit) {
    SDL_SetWindowTitle(window.get(),"Particle World | Resident basin");
    auto materials=loadMaterials(assets/"materials/default.json");
    int seed=1337;auto basin=std::make_unique<BasinPublication>(assets,materials,std::uint32_t(seed));
    PreviewGui gui(window.get());ImGuiRenderer renderer(assets);
    float yaw=.65f,distance=43,elevation=30;bool running=true;std::uint32_t frames=0;
    int width=0,height=0;std::string message;
    std::unique_ptr<VoxelProbes> probes;FixedStep clock;auto previous=std::chrono::steady_clock::now();std::uint32_t completed=0;bool paused=false;int spawnCount=256;float brushCentre[]{-6,6,-4.5f},brushHalf=.5f,brushVelocity[]{2,0,0};
    std::unique_ptr<PendingDebrisTransaction> excavation;
    std::unique_ptr<StagedExplosion> blast;
    std::vector<terrain::ExplosionMaterial> blastMaterials;
    for(const auto& m:materials.entries) {
        const double resistance=m.name=="Sand"?100:m.name=="Wood"?300:m.name=="Ash"?50:1000;
        blastMaterials.push_back({m.gpu.mechanical[0],m.gpu.identity[1]==0,resistance,m.gpu.thermal[3]>0?1.f:0.f});
    }
    float blastCentre[]{-8,3.875f,-4},blastRadius=.5f,blastSpeed=2,blastEnergy=2000;

    std::vector<terrain::ExcavationMaterial> excavationMaterials;
    for(const auto& material:materials.entries)excavationMaterials.push_back({material.gpu.mechanical[0],material.gpu.identity[1]==0,material.gpu.thermal[3]>0?1.f:0.f});
    float excavationCentre[]{-8,3.875f,-4},excavationHalf=.25f,debrisLift=2;
    std::uint64_t excavationSequence=0;std::uint32_t proposedVoxels=0,proposedGrains=0,eligibleVoxels=0;
    std::uint32_t sand=0;for(unsigned i=0;i<materials.entries.size();++i)if(materials.entries[i].name=="Sand")sand=i;
    std::uint32_t brushMaterial=sand;
    std::cout<<"Resident basin: "<<basin->store->occupiedVoxelCount()<<" voxels, "<<basin->mesh->faceCount()<<" exposed faces, seed "<<seed<<"; terrain inspection with optional collision probes\n";
    while(running) {
        if(probes)probes->poll(completed);
        if(excavation) {
            excavation->poll(completed);
            if(excavation->state()==PendingDebrisTransaction::State::Ready) {
                if(probes && excavation->publish(0,*probes,basin->publication)) {
                    basin->refresh();message="Excavated "+std::to_string(proposedVoxels)+" voxels into "+std::to_string(proposedGrains)+" grains. "+std::to_string(eligibleVoxels-proposedVoxels)+" voxels remain in the brush.";
                    std::cout<<"Terrain excavation committed: "<<proposedVoxels<<" voxels -> "<<proposedGrains<<" grains; revision "<<basin->store->revision()<<std::endl;
                }else message="The scene changed; excavation was discarded.";
                excavation.reset();clock.reset();
            }else if(excavation->state()==PendingDebrisTransaction::State::Rejected) {
                const auto errors=excavation->errors();
                message=(errors&32)?"Physics fault: reset the pool before excavating.":(errors&1)?"Not enough particle slots. Terrain was kept.":(errors&8)?"Debris space is occupied. Let grains move or adjust the brush; terrain was kept.":"Debris placement was rejected. Terrain was kept.";
                std::cout<<"Terrain excavation rejected: "<<errors<<std::endl;excavation.reset();clock.reset();
            }
        }
        if(blast && probes) {
            try {if(blast->pump(completed,*probes,basin->publication)) {
                basin->refresh();clock.reset();message="Blast batch committed.";
                std::cout<<"Blast batch "<<blast->batches()<<": "<<blast->removedVoxels()<<" voxels total; "<<blast->remainingJoules()<<" J remaining"<<std::endl;
            }}catch(const std::exception& error){blast->cancel();message=error.what();}
        }
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
        ImGui::SetNextWindowPos({16,16},ImGuiCond_FirstUseEver);ImGui::SetNextWindowSize({420,690},ImGuiCond_FirstUseEver);
        ImGui::Begin("Resident basin");
        ImGui::TextUnformatted("32 x 16 x 32 metres | 0.125 m voxels");
        ImGui::TextWrapped("Add sand or excavate terrain. Particle brushes run on the next tick. Water and burning are not enabled.");
        if(ImGui::CollapsingHeader("Camera and world")) {
        ImGui::TextWrapped("Stone ledge, sand slope, dry pool recess and wood fuel patch.");
        ImGui::SliderAngle("Orbit",&yaw,-180,180);ImGui::SliderFloat("Distance",&distance,18,65,"%.1f m");ImGui::SliderFloat("Height",&elevation,5,50,"%.1f m");
        ImGui::InputInt("Seed",&seed);seed=std::max(0,seed);
        if(ImGui::Button("Rebuild basin")) {
            try {auto candidate=std::make_unique<BasinPublication>(assets,materials,std::uint32_t(seed));blast.reset();excavation.reset();probes.reset();basin=std::move(candidate);clock.reset();message="Basin rebuilt";}
            catch(const std::exception& error){message=error.what();}
        }
        }
        ImGui::Separator();
        ImGui::SliderInt("Grains per brush",&spawnCount,1,1024);
        ImGui::SliderFloat3("Brush centre",brushCentre,-15,15,"%.2f m");
        ImGui::SliderFloat("Brush half width",&brushHalf,.125f,2,"%.3f m");
        ImGui::SliderFloat3("Velocity",brushVelocity,-6,6,"%.1f m/s");
        auto queueBrush=[&](CommandType type,bool all=false) {
            Command command;command.type=type;command.material=type==CommandType::SpawnBrush?sand:brushMaterial;command.count=type==CommandType::SpawnBrush?std::uint32_t(spawnCount):0;
            for(unsigned axis=0;axis<3;++axis){command.minimum[axis]=all?-16:std::max(-16.0f,brushCentre[axis]-brushHalf);command.maximum[axis]=all?16:std::min(16.0f,brushCentre[axis]+brushHalf);command.parameters[axis]=brushVelocity[axis];}
            const auto reason=probes->enqueue(command);message=reason.empty()?(paused?"Command queued; resume to apply":"Command queued for next tick"):std::string(reason);
        };
        if(ImGui::Button("Add sand")) {
            try {if(!probes || !probes->hasPool()){auto candidate=std::make_unique<VoxelProbes>(assets,materials,true,true);probes=std::move(candidate);clock.reset();}queueBrush(CommandType::SpawnBrush);}
            catch(const std::exception& error){message=error.what();}
        }
        if(probes && probes->hasPool()) {
            ImGui::SetNextItemWidth(220);
            if(ImGui::BeginCombo("Particle material",materials.entries[brushMaterial].name.c_str())) {
                for(unsigned m=0;m<materials.entries.size();++m)if(materials.entries[m].gpu.identity[1]==0) {
                    if(ImGui::Selectable(materials.entries[m].name.c_str(),brushMaterial==m))brushMaterial=m;
                }
                ImGui::EndCombo();
            }
            const auto& target=materials.entries[brushMaterial].name;
            if(ImGui::Button(("Clear "+target+"##ClearTarget").c_str()))queueBrush(CommandType::RemoveBrush,true);
            ImGui::SameLine();if(ImGui::Button(("Remove "+target+"##RemoveTarget").c_str()))queueBrush(CommandType::RemoveBrush);
            ImGui::SameLine();if(ImGui::Button(("Impulse "+target+"##ImpulseTarget").c_str()))queueBrush(CommandType::ApplyImpulse);
            const auto& pool=probes->poolSnapshot();
            if(pool.valid)ImGui::Text("Particles: %u / %u | rejected: %u",pool.active,probes->capacity(),pool.rejected);
            ImGui::Text("Queued commands: %zu",probes->pendingCommands());
        }
        if(ImGui::CollapsingHeader("Terrain editing",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Terrain centre",excavationCentre,-15,15,"%.3f m");
            ImGui::SliderFloat("Cut half width",&excavationHalf,.125f,1,"%.3f m");
            ImGui::SliderFloat("Debris lift",&debrisLift,0,6,"%.1f m/s");
            ImGui::TextWrapped("Excavate up to 32 voxels per click. Terrain stays intact if debris cannot fit. Unsupported overhangs stay static.");
            ImGui::BeginDisabled(bool(excavation) || (blast && blast->active()));
            if(ImGui::Button("Excavate brush")) {
                try {
                    if(probes && probes->hasPool() && probes->pendingCommands())throw std::invalid_argument("Let queued particle commands run before excavating.");
                    if(probes && probes->fault())throw std::invalid_argument("Reset the particle pool after its physics fault.");
                    Command command;command.type=CommandType::EditTerrain;command.tick=probes && probes->hasPool()?probes->ticks():0;command.sequence=excavationSequence;command.count=32;command.parameters[1]=debrisLift;
                    for(unsigned a=0;a<3;++a){command.minimum[a]=std::max(-16.f,excavationCentre[a]-excavationHalf);command.maximum[a]=std::min(16.f,excavationCentre[a]+excavationHalf);}
                    const auto plan=terrain::planExcavation(*basin->store,command,command.tick,excavationMaterials,VoxelProbes::poolCapacity);
                    if(plan.state==terrain::ExcavationPlan::State::Empty)message="No voxel centres inside this brush.";
                    else if(plan.state==terrain::ExcavationPlan::State::NoSpace)message="No nearby room for the required debris. Terrain was kept.";
                    else if(plan.state==terrain::ExcavationPlan::State::NoCapacity)message="The required debris exceeds the batch capacity. Terrain was kept.";
                    else {
                        if(!probes || !probes->hasPool()) {
                            auto candidate=std::make_unique<VoxelProbes>(assets,materials,true,true);
                            auto transaction=std::make_unique<PendingDebrisTransaction>(assets,*candidate,basin->publication,plan.removals,plan.seeds,materials);
                            probes=std::move(candidate);excavation=std::move(transaction);clock.reset();
                        }else excavation=std::make_unique<PendingDebrisTransaction>(assets,*probes,basin->publication,plan.removals,plan.seeds,materials);
                        excavation->submit(8,*probes);++excavationSequence;
                        proposedVoxels=std::uint32_t(plan.removals.size());proposedGrains=std::uint32_t(plan.seeds.size());eligibleVoxels=plan.eligible;
                        message="Preparing excavation...";
                    }
                }catch(const std::exception& error){excavation.reset();message=error.what();}
            }
            ImGui::EndDisabled();
            if(excavation){ImGui::SameLine();if(ImGui::Button("Cancel excavation")){excavation.reset();clock.reset();message="Excavation cancelled; terrain kept.";}}
            if(probes)for(unsigned m=0;m<probes->debrisMassTotals().size();++m) {
                const auto& mass=probes->debrisMassTotals()[m];if(!mass.removedVoxels)continue;
                ImGui::Text("%s converted: %llu voxels -> %llu grains",materials.entries[m].name.c_str(),static_cast<unsigned long long>(mass.removedVoxels),static_cast<unsigned long long>(mass.activatedGrains));
                ImGui::Text("Mass: %.3f -> %.3f kg (delta %+.3f)",mass.sourceKilograms,mass.destinationKilograms,mass.quantizationKilograms);
            }
        }
        if(ImGui::CollapsingHeader("Explosion")) {
            ImGui::SliderFloat3("Blast centre",blastCentre,-15,15,"%.3f m");blastCentre[1]=std::max(0.f,blastCentre[1]);
            ImGui::SliderFloat("Blast radius",&blastRadius,.125f,2,"%.3f m");
            ImGui::SliderFloat("Debris speed",&blastSpeed,0,12,"%.1f m/s");
            ImGui::SliderFloat("Terrain energy",&blastEnergy,10,10000,"%.0f J");
            ImGui::TextWrapped("Converts terrain in batches of up to 32 voxels. Energy is charged after each successful commit.");
            ImGui::BeginDisabled(bool(excavation) || (blast && blast->active()));
            if(ImGui::Button("Start blast")) {
                try {
                    if(probes && probes->fault())throw std::invalid_argument("Reset the particle pool after its physics fault.");
                    terrain::ExplosionRequest request;request.centre={blastCentre[0],blastCentre[1],blastCentre[2]};request.radius=blastRadius;request.impulse=blastSpeed;request.energyJoules=blastEnergy;request.sequence=excavationSequence;
                    if(!probes || !probes->hasPool()) {
                        auto candidate=std::make_unique<VoxelProbes>(assets,materials,true,true);
                        auto command=std::make_unique<StagedExplosion>(assets,*candidate,basin->publication,materials,request,blastMaterials);
                        probes=std::move(candidate);blast=std::move(command);clock.reset();
                    }else {request.tick=probes->ticks();blast=std::make_unique<StagedExplosion>(assets,*probes,basin->publication,materials,request,blastMaterials);}
                    ++excavationSequence;blast->pump(completed,*probes,basin->publication);message="Blast queued.";
                }catch(const std::exception& error){if(blast)blast->cancel();message=error.what();}
            }
            ImGui::EndDisabled();
            if(blast) {
                if(blast->active()){ImGui::SameLine();if(ImGui::Button("Cancel blast")){blast->cancel();clock.reset();message="Blast cancelled; committed batches retained.";}}
                const char* status="";
                switch(blast->state()) {
                case StagedExplosion::State::WaitingForTick:status="Waiting for next simulation tick";break;
                case StagedExplosion::State::WaitingForGpu:status="Validating batch";break;
                case StagedExplosion::State::Deferred:status=(blast->errors()&1)?"Waiting for particle capacity":"Waiting for debris clearance";break;
                case StagedExplosion::State::Complete:status=blast->completion()==StagedExplosion::Completion::Empty?"Complete":blast->completion()==StagedExplosion::Completion::Energy?"Stopped: no affordable voxels":"Stopped: product quota";break;
                case StagedExplosion::State::NoSpace:status="Stopped: no room for required debris";break;
                case StagedExplosion::State::Cancelled:status="Cancelled";break;
                case StagedExplosion::State::Stale:status="Stopped: scene replaced";break;
                case StagedExplosion::State::Faulted:status="Stopped: validation failed";break;
                }
                ImGui::TextWrapped("%s",status);
                ImGui::Text("Blast: %llu batches | %llu voxels",static_cast<unsigned long long>(blast->batches()),static_cast<unsigned long long>(blast->removedVoxels()));
                ImGui::Text("Energy: %.1f J left | %.1f J charged",blast->remainingJoules(),blast->chargedJoules());
            }
        }
        if(ImGui::Button("Reset particle pool")){try{auto candidate=std::make_unique<VoxelProbes>(assets,materials,true,true);blast.reset();excavation.reset();probes=std::move(candidate);clock.reset();message="Empty particle pool; counters and faults reset";}catch(const std::exception& error){message=error.what();}}
        if(ImGui::Button("Drop interacting diagnostics")){try{auto candidate=std::make_unique<VoxelProbes>(assets,materials,true);blast.reset();excavation.reset();probes=std::move(candidate);clock.reset();paused=false;message="256 fixed diagnostic grains";}catch(const std::exception& error){message=error.what();}}
        if(ImGui::Button("Drop collision probes")){try{auto candidate=std::make_unique<VoxelProbes>(assets,materials);blast.reset();excavation.reset();probes=std::move(candidate);clock.reset();paused=false;message="256 terrain probes; no grain-to-grain contacts";}catch(const std::exception& error){message=error.what();}}
        if(probes){ImGui::Checkbox("Pause simulation",&paused);ImGui::Text("Simulation tick: %llu",static_cast<unsigned long long>(probes->ticks()));if(probes->hasSettling()) {
            const auto& settling=probes->settlingSnapshot();
            if(settling.valid){ImGui::Text("Settling candidates: %u / %u",settling.free,probes->capacity());if(settling.active)ImGui::TextWrapped("Settling tracker fault. Reset the particle pool to recover.");}
        }if(probes->fault())ImGui::TextWrapped("Physics fault (phase mask %u): the tick was rolled back. Reset the particle pool to recover.",probes->errors());}
        ImGui::Text("%llu occupied voxels | %u faces",static_cast<unsigned long long>(basin->store->occupiedVoxelCount()),basin->mesh->faceCount());
        ImGui::Text("Terrain GPU payload: %.1f MiB",double(basin->occupancy->payloadBytes()+basin->mesh->payloadBytes())/1048576);
        if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
        ImGui::End();
        const auto batch=clock.advance(elapsed,!probes || paused || probes->fault() || probes->transactionPending(),false);
        // Sample by rendered frame: a four-tick catch-up cadence can skip every
        // tick divisible by eight and leave old diagnostics visible indefinitely.
        if(probes){for(unsigned tick=0;tick<batch.count;++tick)probes->tick(std::uint16_t(tick+1),*basin->occupancy,*basin->store);probes->prepare(5);if(paused || frames%8==0)probes->request(6);}
        float view[16],projection[16];
        bx::mtxLookAt(view,{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance},{0,1,0});
        bx::mtxProj(projection,50.0f,float(width)/float(height),.1f,200.0f,bgfx::getCaps()->homogeneousDepth);
        bgfx::setViewRect(10,0,0,std::uint16_t(width),std::uint16_t(height));
        bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x182333ff,1.0f,0);bgfx::setViewTransform(10,view,projection);bgfx::touch(10);
        basin->mesh->render(10,*basin->store,*basin->occupancy,{});
        if(probes)probes->render(10);
        ImGui::Render();renderer.render(ImGui::GetDrawData(),20);completed=bgfx::frame();
        ++frames;if(frameLimit && frames>=frameLimit)break;
    }
    return 0;
}
}
