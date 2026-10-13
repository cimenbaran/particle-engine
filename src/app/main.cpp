#include "core/ComputeDispatch.hpp"
#include "platform/Window.hpp"
#include "Benchmark.hpp"
#include "ResourceUsage.hpp"
#include "core/RunOptions.hpp"
#include "core/FixedStep.hpp"
#include "core/FlyCamera.hpp"
#include "core/PointerBrush.hpp"
#include "core/CommandQueue.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include "assets/Scenes.hpp"
#include "assets/RecordingSession.hpp"
#include "render/ImGuiRenderer.hpp"
#include "render/Shaders.hpp"
#include "render/Capabilities.hpp"
#include "render/GpuLayoutTest.hpp"
#include "render/GpuPrefixScan.hpp"
#include "render/GpuRadixSort.hpp"
#include "render/GpuSpatialKeys.hpp"
#include "render/GpuCellRanges.hpp"
#include "render/GpuCellHash.hpp"
#include "render/GpuNeighborsTest.hpp"
#include "render/GpuJacobiContacts.hpp"
#include "render/GpuVelocityReconstruction.hpp"
#include "render/GpuSweptBounds.hpp"
#include "render/GpuVoxelSweep.hpp"
#include "render/GpuVoxelWorld.hpp"
#include "render/GpuGranularStep.hpp"
#include "render/GpuSettling.hpp"
#include "render/SphereTerrain.hpp"
#include "render/GpuParticleDebug.hpp"
#include "render/GpuContactNormals.hpp"
#include "render/GpuGranularStability.hpp"
#include "render/GpuContactVelocity.hpp"
#include "render/ParticlePool.hpp"
#include "render/PoolDiagnostics.hpp"
#include "render/ParticleVisibility.hpp"
#include "render/ParticleAudit.hpp"
#include "render/GpuVisibilityTest.hpp"
#include "render/GpuPoolTest.hpp"
#include <memory>
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
#include <cstdio>
#include <cstring>
#include <fstream>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#endif

namespace {
struct Graphics {
    Graphics(pw::Window& window,bool vsync=true,bool software=false) {
        int w, h; window.drawable(w,h);
        bgfx::Init init;
#if defined(__APPLE__)
        if(software) throw std::runtime_error("--software-gpu requires Windows D3D11/WARP");
        init.type = bgfx::RendererType::Metal;
#else
        init.type = bgfx::RendererType::Direct3D11;
        auto releaseDevice=[](ID3D11Device* device) {if(device) device->Release();};
        std::unique_ptr<ID3D11Device,decltype(releaseDevice)> softwareDevice(nullptr,releaseDevice);
        if(software) {
            // Supply a real WARP device: the pinned adapter enumeration can
            // override its software vendor hint while choosing adapter zero.
            ID3D11Device* device=nullptr;const D3D_FEATURE_LEVEL feature=D3D_FEATURE_LEVEL_11_0;
            const HRESULT result=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,&feature,1,D3D11_SDK_VERSION,&device,nullptr,nullptr);
            softwareDevice.reset(device);
            if(FAILED(result)) throw std::runtime_error("Cannot create D3D11 WARP feature-level 11 device");
            init.platformData.context=device; // bgfx retains its own COM reference.
        }
#endif
        init.swapChain.nwh = window.nativeHandle();
        init.fallback = false;
        init.profile = true;
        init.swapChain.width = uint32_t(w); init.swapChain.height = uint32_t(h);
        init.reset = vsync?BGFX_RESET_VSYNC:0;
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
    const uint32_t particleCount;
    std::filesystem::path assetsPath;
    std::unique_ptr<pw::GpuGranularStep> granular;
    std::unique_ptr<pw::PoolDiagnostics> granularDiagnostics,neighborhoodDiagnostics,occupancyDiagnostics,totalDiagnostics;
    std::unique_ptr<pw::GpuSettling> settling;
    std::unique_ptr<pw::SphereTerrain> terrain;
    std::unique_ptr<pw::GpuParticleDebug> debugView;
    std::unique_ptr<pw::GpuContactNormals> contactNormals;
    int debugMode=0;
    std::unique_ptr<pw::PoolDiagnostics> settlingDiagnostics;
    bool granularFault=false;
    std::string solverMessage;
    bgfx::DynamicVertexBufferHandle state[2]{BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE};
    bgfx::DynamicVertexBufferHandle instances = BGFX_INVALID_HANDLE;
    bgfx::VertexBufferHandle vertices = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle compute = BGFX_INVALID_HANDLE, initialize = BGFX_INVALID_HANDLE, draw = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uniform = BGFX_INVALID_HANDLE;
    unsigned current{};
    pw::CommandQueue commands;
    std::string commandMessage;
    std::unique_ptr<pw::ParticlePool> pool;
    std::unique_ptr<pw::PoolDiagnostics> diagnostics;
    std::unique_ptr<pw::ParticleVisibility> visibility;
    std::unique_ptr<pw::PoolDiagnostics> visibleDiagnostics;
    pw::MaterialLibrary materials;
    bgfx::DynamicVertexBufferHandle materialBuffer=BGFX_INVALID_HANDLE;
    std::filesystem::path materialPath;
    std::string materialMessage;
    std::vector<std::uint32_t> liveMaterialIds;
    int sceneMaterial=-1; // -1 mixes the available solid profiles.
    bool reloadPending=false;
    pw::ScenePreset scene;
    pw::RecordingSession session;
    std::filesystem::path scenePath;
    std::string sceneMessage;
    explicit Particles(const std::filesystem::path& assets,uint32_t capacity,const std::filesystem::path& preset={},const pw::ScenePreset* initialScene=nullptr):particleCount(capacity),assetsPath(assets) {
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
            materialPath=assets/"materials"/"default.json";
            materials=pw::loadMaterials(materialPath);
            if(initialScene) scene=*initialScene;
            if(!preset.empty()) {scene=pw::loadScene(preset);scenePath=preset;}
            pw::validateSceneCapacity(scene,particleCount);pw::sceneMaterials(scene,materials);
            uploadMaterials(materials);
            pool=std::make_unique<pw::ParticlePool>(assets,particleCount);
            diagnostics=std::make_unique<pw::PoolDiagnostics>(assets);
            visibility=std::make_unique<pw::ParticleVisibility>(assets,particleCount);
            visibleDiagnostics=std::make_unique<pw::PoolDiagnostics>(assets);
            reset();
        } catch (...) { release(); throw; }
    }
    void validateGranularScene(const pw::ScenePreset& candidate) const {
        for(unsigned axis=0;axis<3;++axis) if(candidate.minimum[axis]<(axis==1?0.0625f:-7.9375f) || candidate.maximum[axis]>(axis==1?15.9375f:7.9375f))
            throw std::runtime_error("Granular scene centres must fit inside the radius-inset enclosure");
    }
    void setGranular(bool enabled) {
        if(session.active()) throw std::runtime_error("Finish the recording session before changing solver");
        if(!enabled && scene.obstacle.radius>0)throw std::runtime_error("Load a scene without terrain before disabling particle collisions");
        if(enabled) {
            validateGranularScene(scene);
            auto candidate=std::make_unique<pw::GpuGranularStep>(assetsPath,particleCount,pw::GranularStepProfile{},4,9.81f,true,scene.obstacle);
            auto diagnostic=std::make_unique<pw::PoolDiagnostics>(assetsPath);
            auto tracker=std::make_unique<pw::GpuSettling>(assetsPath,particleCount,pw::SettlingProfile{},true,scene.obstacle);
            auto occupancyDiagnostic=std::make_unique<pw::PoolDiagnostics>(assetsPath);
            auto totalDiagnostic=std::make_unique<pw::PoolDiagnostics>(assetsPath);
            auto neighborhoodDiagnostic=std::make_unique<pw::PoolDiagnostics>(assetsPath);
            auto trackingDiagnostic=std::make_unique<pw::PoolDiagnostics>(assetsPath);
            auto debug=std::make_unique<pw::GpuParticleDebug>(assetsPath);
            auto surface=scene.obstacle.radius>0?std::make_unique<pw::SphereTerrain>(assetsPath):nullptr;
            terrain=std::move(surface);granular=std::move(candidate);granularDiagnostics=std::move(diagnostic);
            totalDiagnostics=std::move(totalDiagnostic);occupancyDiagnostics=std::move(occupancyDiagnostic);
            debugView=std::move(debug);neighborhoodDiagnostics=std::move(neighborhoodDiagnostic);
            settling=std::move(tracker);settlingDiagnostics=std::move(trackingDiagnostic);
        } else {terrain.reset();debugView.reset();contactNormals.reset();debugMode=0;settlingDiagnostics.reset();settling.reset();totalDiagnostics.reset();occupancyDiagnostics.reset();neighborhoodDiagnostics.reset();granularDiagnostics.reset();granular.reset();}
        granularFault=false;reloadPending=false;reset();solverMessage.clear();
    }
    void stageShaders(pw::ShaderReload& reload) {
        if(granular) throw std::runtime_error("Switch off particle collisions before shader reload");
        reload.compute(compute,"cs_particles");reload.compute(initialize,"cs_reset");
        reload.graphics(draw,"vs_particles","fs_particles");
        pool->stageShaders(reload);diagnostics->stageShaders(reload);
        visibility->stageShaders(reload);visibleDiagnostics->stageShaders(reload);
    }
    void uploadMaterials(const pw::MaterialLibrary& candidate) {
        const auto packed=candidate.packed();
        bgfx::VertexLayout lanes;
        lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        auto replacement=bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),uint32_t(packed.size()*sizeof(pw::MaterialGpu))),lanes,BGFX_BUFFER_COMPUTE_READ);
        if(!bgfx::isValid(replacement)) throw std::runtime_error("Material GPU allocation failed");
        if(bgfx::isValid(materialBuffer)) bgfx::destroy(materialBuffer);
        materialBuffer=replacement;
    }
    void reloadAtTickBoundary() {
        if(!reloadPending) return;
        reloadPending=false;
        try {
            auto candidate=pw::loadMaterials(materialPath);
            pw::validateMaterialReload(materials,candidate,liveMaterialIds);
            pw::sceneMaterials(scene,candidate); // Keep the stored preset resettable.
            uploadMaterials(candidate);
            materials=std::move(candidate);
            materialMessage="Materials reloaded";
        } catch(const std::exception& error) {materialMessage=std::string("Reload rejected: ")+error.what();}
    }
    void release() {
        debugView.reset();contactNormals.reset();debugMode=0;settlingDiagnostics.reset();settling.reset();totalDiagnostics.reset();occupancyDiagnostics.reset();neighborhoodDiagnostics.reset();granularDiagnostics.reset();granular.reset();
        visibleDiagnostics.reset();
        diagnostics.reset();
        visibility.reset();
        pool.reset();
        if(bgfx::isValid(materialBuffer)) bgfx::destroy(materialBuffer);
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
    void reset(uint32_t initialCount=UINT32_MAX) {
        auto ids=sceneMaterial<0?pw::sceneMaterials(scene,materials):std::vector<uint32_t>{uint32_t(sceneMaterial)};
        auto initial=pw::instantiateScene(scene,materials,particleCount,ids,
            initialCount==UINT32_MAX?std::nullopt:std::optional<uint32_t>(initialCount));

        if(granular) {
            validateGranularScene(scene);
            for(auto& p:initial) if(p.identity[1]&1) {
                p.positionRadius[3]=0.0625f;
                p.velocityInvMass[3]=1.0f/(4.0f/3.0f*3.14159265f*0.0625f*0.0625f*0.0625f*materials.entries[p.identity[0]].gpu.mechanical[0]);
            }
            granular->resetFailures(0);granularDiagnostics->beginEpoch();neighborhoodDiagnostics->beginEpoch();occupancyDiagnostics->beginEpoch();totalDiagnostics->beginEpoch();
            settling->reset(0);settlingDiagnostics->beginEpoch();granularFault=false;
        }
        diagnostics->beginEpoch();
        visibleDiagnostics->beginEpoch();
        bgfx::setViewMode(0,bgfx::ViewMode::Sequential);
        commands.reset(); commandMessage.clear();
        for (auto& h: state) { if(bgfx::isValid(h)) bgfx::destroy(h); h = BGFX_INVALID_HANDLE; }
        if(bgfx::isValid(instances)) bgfx::destroy(instances);
        instances = BGFX_INVALID_HANDLE;
        liveMaterialIds=std::move(ids);
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
        const float params[]{0,float(particleCount),float(materials.entries.size()),0};
        bgfx::setUniform(uniform,params);
        bgfx::setBuffer(0,upload,bgfx::Access::Read);
        bgfx::setBuffer(1,state[0],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer,bgfx::Access::Read);
        bgfx::dispatch(0,initialize,pw::dispatchGroups(particleCount));
        bgfx::destroy(upload); // bgfx defers destruction past submitted work.
        current=0;
        pool->rebuild(0,state[current],true);
    }
    bool loadPreset(const std::filesystem::path& path) {
        std::optional<pw::ScenePreset> candidate;
        try {
            candidate=pw::loadScene(path);pw::validateSceneCapacity(*candidate,particleCount);
            pw::sceneMaterials(*candidate,materials);
            if(granular) validateGranularScene(*candidate);
        } catch(const std::exception& error) {sceneMessage=std::string("Scene rejected: ")+error.what();return false;}
        scene=std::move(*candidate);scenePath=path;sceneMaterial=-1;
        if(granular || scene.obstacle.radius>0)setGranular(true);else reset();
        sceneMessage="Scene loaded";return true;
    }
    void captureFromReset() {
        if(granular) throw std::runtime_error("Granular recording profile is not implemented yet");
        auto starting=scene;
        if(sceneMaterial>=0) starting.materials={materials.entries[sceneMaterial].name};
        session.capture(starting,materials,particleCount,pw::workgroupSize);
        reloadPending=false;reset();
    }
    void beginReplay(pw::CommandRecording recording,bool resetScene=true) {
        if(granular) throw std::runtime_error("Switch off particle collisions before ballistic replay");
        session.replay(std::move(recording),materials,particleCount);
        scene=session.recording().scene;sceneMaterial=-1;scenePath.clear();reloadPending=false;
        if(resetScene) reset();
    }
    void queueBrush(pw::CommandType type,float strength,uint32_t count=0,float lifetime=0,
                    const pw::BrushBox* bounds=nullptr,uint32_t material=UINT32_MAX) {
        if(session.replaying()) {commandMessage="Brush input is disabled during replay";return;}
        if(session.capturing() && session.recording().commands.size()+commands.pending()>=pw::maxRecordingCommands) {
            commandMessage="Recording command quota reached";return;
        }
        // Fixed-box controls use the scene override or the preset's first material.
        pw::Command command;
        command.tick=commands.nextTick(); command.type=type;
        command.minimum={-4,0,-4}; command.maximum={4,12,4};
        command.parameters={0,strength,0,lifetime}; command.count=count;
        if(type==pw::CommandType::SpawnBrush) {command.minimum={-2,6,-2};command.maximum={2,10,2};}
        if(bounds) {command.minimum=bounds->minimum;command.maximum=bounds->maximum;}
        command.material=material==UINT32_MAX?(sceneMaterial<0?liveMaterialIds.front():unsigned(sceneMaterial)):material;
        if(granular && type==pw::CommandType::SpawnBrush) {
            if(command.material>=materials.entries.size() || materials.entries[command.material].gpu.identity[1]!=0) {commandMessage="Granular spawns require a solid material";return;}
            for(unsigned axis=0;axis<3;++axis) {
                command.minimum[axis]=std::max(command.minimum[axis],axis==1?0.0625f:-7.9375f);
                command.maximum[axis]=std::min(command.maximum[axis],axis==1?15.9375f:7.9375f);
                if(command.minimum[axis]>command.maximum[axis]) {commandMessage="Spawn brush is outside the granular enclosure";return;}
            }
        }
        if(granular && type==pw::CommandType::SpawnBrush && scene.obstacle.radius>0) {
            double distance=0;for(unsigned a=0;a<3;++a){double d=double(scene.obstacle.centre[a])-std::clamp(scene.obstacle.centre[a],command.minimum[a],command.maximum[a]);distance+=d*d;}
            const double radius=double(scene.obstacle.radius)+0.0625;
            if(distance<=radius*radius){commandMessage="Spawn brush intersects sphere terrain; move the brush outside it";return;}
        }
        pw::CommandLimits limits; limits.materialCount=uint32_t(materials.entries.size());
        const auto reason=commands.enqueue(command,limits);
        // Reload runs before queued commands are consumed. Reserve newly selected
        // spawn materials immediately and conservatively until the next reset.
        if(reason.empty() && type==pw::CommandType::SpawnBrush && count &&
           std::find(liveMaterialIds.begin(),liveMaterialIds.end(),command.material)==liveMaterialIds.end())
            liveMaterialIds.push_back(command.material);
        commandMessage=reason.empty()?"Brush queued for next tick":std::string(reason);
    }
    void tick(uint16_t view) {
        if(!session.active()) reloadAtTickBoundary();
        if(session.replaying()) for(const auto& command:session.due(commands.nextTick())) {
            pw::CommandLimits limits;limits.materialCount=uint32_t(materials.entries.size());
            if(auto reason=commands.enqueueRecorded(command,limits);!reason.empty()) throw std::runtime_error("Replay queue rejected a validated command: "+std::string(reason));
            if(command.type==pw::CommandType::SpawnBrush && std::find(liveMaterialIds.begin(),liveMaterialIds.end(),command.material)==liveMaterialIds.end())
                liveMaterialIds.push_back(command.material);
        }
        bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
        const auto batch=commands.consume();
        for(const auto& command:batch) {
            pw::CommandLimits limits;limits.materialCount=uint32_t(materials.entries.size());
            if(!pw::validate(command,limits,command.tick).empty()) {
                if(session.active()) throw std::runtime_error("Recording command rejected before GPU submission");
                commandMessage="Queued command rejected after material reload";continue;
            }
            pool->apply(view,state[current],materialBuffer,uint32_t(materials.entries.size()),command,granular?0.0625f:0.045f);
            commandMessage="Brush submitted (spawns limited by free capacity)";
        }
        const float params[]{float(pw::FixedStep::tickSeconds),float(particleCount),float(materials.entries.size()),0};
        auto source=state[current];auto program=compute;
        if(granular) {granular->submit(view,source,materialBuffer,uint32_t(materials.entries.size()));source=granular->output();
            settling->submit(view,source,materialBuffer,uint32_t(materials.entries.size()),granular->status());program=initialize;}
        bgfx::setUniform(uniform, params);
        bgfx::setBuffer(0,source,bgfx::Access::Read);
        bgfx::setBuffer(1,state[1-current],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer,bgfx::Access::Read);
        bgfx::dispatch(view,program,pw::dispatchGroups(particleCount));
        current=1-current;
        pool->rebuild(view,state[current]);
        session.observed(batch,commands.nextTick());
    }
    void prepareDebug(uint16_t view) {
        if(!debugView)return;
        if(debugMode==int(pw::ParticleDebugMode::ContactNormal)) {
            if(!contactNormals)contactNormals=std::make_unique<pw::GpuContactNormals>(assetsPath,particleCount);
            contactNormals->submit(view,state[current],materialBuffer,uint32_t(materials.entries.size()),scene.obstacle);
        }
        debugView->submit(view,particleCount,pw::ParticleDebugMode(debugMode),state[current],materialBuffer,
            uint32_t(materials.entries.size()),settling->history(),settling->status(),instances,
            contactNormals?contactNormals->output():bgfx::DynamicVertexBufferHandle BGFX_INVALID_HANDLE,granular->status());
    }
    void render(uint16_t view,bool indirect) {
        if(terrain)terrain->render(view,scene.obstacle);
        bgfx::setVertexBuffer(0,vertices); bgfx::setIndexBuffer(indices);
        bgfx::setInstanceDataBuffer(visibility->instances(),0,particleCount);
        bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);
        if(indirect && visibility->supportsIndirect()) bgfx::submit(view,draw,visibility->arguments(),0,1);
        else bgfx::submit(view,draw);
    }
};
// Explicit diagnostic readback only. Normal simulation and successful soaks do
// not allocate or synchronize through this path. The export captures a prefix.
std::vector<pw::ParticleGpu> snapshotParticles(Particles& particles,const std::filesystem::path& assets,
    bgfx::DynamicVertexBufferHandle buffer=BGFX_INVALID_HANDLE,unsigned lanes=4) {
    const auto count=std::min(particles.particleCount,uint32_t(4096));
    if(!count)return {};
    struct Export {
        bgfx::ProgramHandle program=BGFX_INVALID_HANDLE;
        bgfx::TextureHandle output=BGFX_INVALID_HANDLE,staging=BGFX_INVALID_HANDLE;
        ~Export() {if(bgfx::isValid(program)) bgfx::destroy(program);if(bgfx::isValid(output)) bgfx::destroy(output);if(bgfx::isValid(staging)) bgfx::destroy(staging);}
    } resources;
    resources.program=pw::loadCompute(assets,"cs_export");
    resources.output=bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE);
    resources.staging=bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    if(!bgfx::isValid(resources.output) || !bgfx::isValid(resources.staging)) throw std::runtime_error("Particle snapshot readback allocation failed");
    const float params[]{0,float(count),0,float(lanes)};
    bgfx::setUniform(particles.uniform,params);bgfx::setBuffer(0,bgfx::isValid(buffer)?buffer:particles.state[particles.current],bgfx::Access::Read);
    bgfx::setImage(1,resources.output,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
    bgfx::dispatch(6,resources.program,pw::dispatchGroups(count));
    bgfx::TextureRegion dst;dst.handle=resources.staging;bgfx::TextureRegion src;src.handle=resources.output;
    bgfx::blit(7,dst,src);std::vector<pw::ParticleGpu> result(count);
    const auto ready=bgfx::read(dst,result.data());while(bgfx::frame()<ready) SDL_PumpEvents();
    return result;
}
void runGpuPoolIsolationTest(const std::filesystem::path& assets) {
    pw::ScenePreset effectScene;effectScene.name="Cosmetic isolation test";effectScene.count=0;effectScene.materials={"Wood"};
    Particles physical(assets,129),cosmetic(assets,127,{},&effectScene);
    pw::ParticleAudit physicalAudit(assets,129),cosmeticAudit(assets,127);
    bgfx::frame();
    auto require=[](bool condition,const char* message) {if(!condition) throw std::runtime_error(message);};
    auto snapshot=[&](Particles& particles,bgfx::DynamicVertexBufferHandle buffer=BGFX_INVALID_HANDLE,unsigned lanes=4) {
        return snapshotParticles(particles,assets,buffer,lanes);
    };
    const auto original=snapshot(physical);
    auto check=[&](uint32_t physicalAlive,uint32_t cosmeticAlive,uint32_t dropped) {
        for(auto pair:{std::pair{&physical,&physicalAudit},std::pair{&cosmetic,&cosmeticAudit}}) {
            auto& p=*pair.first;auto& a=*pair.second;
            a.prepare(4,p.state[p.current],p.pool->activeSlots(),p.pool->freeSlots(),p.pool->counts(),uint32_t(p.materials.entries.size()));
            require(a.request(6,p.commands.nextTick()),"Isolation audit request failed");
            require(p.diagnostics->request(6,p.pool->counts(),p.commands.nextTick()),"Isolation counts request failed");
        }
        while(physicalAudit.pending() || cosmeticAudit.pending() || physical.diagnostics->pending() || cosmetic.diagnostics->pending()) {
            const auto frame=bgfx::frame();physicalAudit.poll(frame);cosmeticAudit.poll(frame);
            physical.diagnostics->poll(frame);cosmetic.diagnostics->poll(frame);SDL_PumpEvents();
        }
        for(auto pair:{std::pair{&physical,&physicalAudit},std::pair{&cosmetic,&cosmeticAudit}}) {
            const auto& a=pair.second->latest();
            require(a.valid && a.active==pair.first->particleCount && !a.free && !a.reserved && !a.rejected,"Independent pool audit failed");
        }
        const auto& p=physical.diagnostics->latest();const auto& c=cosmetic.diagnostics->latest();
        require(p.valid && p.active==physicalAlive && p.free==129-physicalAlive && !p.rejected,"Cosmetics consumed physical capacity");
        require(c.valid && c.active==cosmeticAlive && c.free==127-cosmeticAlive && c.rejected==dropped,"Cosmetic capacity/drop accounting failed");
    };
    check(129,0,0);
    cosmetic.queueBrush(pw::CommandType::SpawnBrush,6,128,float(2*pw::FixedStep::tickSeconds));
    cosmetic.tick(1);check(129,127,1);
    cosmetic.tick(1);check(129,0,1);
    cosmetic.queueBrush(pw::CommandType::SpawnBrush,6,1,2);cosmetic.tick(1);check(129,1,1);
    cosmetic.reset(0);check(129,0,0);
    const auto unchanged=snapshot(physical);
    require(std::memcmp(original.data(),unchanged.data(),original.size()*sizeof(pw::ParticleGpu))==0,"Cosmetic operations modified physical state bytes");
    cosmetic.queueBrush(pw::CommandType::SpawnBrush,6,1,2);cosmetic.tick(1);check(129,1,0);
    const auto cosmeticBefore=snapshot(cosmetic);
    physical.reset(0);check(0,1,0);
    const auto cosmeticAfter=snapshot(cosmetic);
    require(std::memcmp(cosmeticBefore.data(),cosmeticAfter.data(),cosmeticBefore.size()*sizeof(pw::ParticleGpu))==0,"Physical reset modified cosmetic state bytes");
    cosmetic.queueBrush(pw::CommandType::SpawnBrush,6,128,2);cosmetic.tick(1);check(0,127,2);
    const auto fullCosmetic=snapshot(cosmetic);
    physical.queueBrush(pw::CommandType::SpawnBrush,0,129,0);physical.tick(1);check(129,127,2);
    const auto stillFullCosmetic=snapshot(cosmetic);
    require(std::memcmp(fullCosmetic.data(),stillFullCosmetic.data(),fullCosmetic.size()*sizeof(pw::ParticleGpu))==0,"Physical spawn modified cosmetic state bytes");
    std::cout<<"GPU physical/cosmetic isolation, independent overflow, expiry, reuse and reset passed\n";
    {
        pw::ScenePreset granularScene;granularScene.count=0;granularScene.materials={"Sand"};
        Particles sandbox(assets,129,{},&granularScene);sandbox.setGranular(true);bgfx::frame();
        sandbox.queueBrush(pw::CommandType::SpawnBrush,0,64);sandbox.tick(1);
        auto spawned=snapshot(sandbox);unsigned alive=0;
        for(const auto& p:spawned)if(p.identity[1]&1){++alive;require(p.positionRadius[3]==0.0625f,"Granular application spawn radius mismatch");}
        require(alive==64,"Granular application spawn count mismatch");
        sandbox.queueBrush(pw::CommandType::ApplyImpulse,30);sandbox.tick(1);
        pw::ParticleAudit audit(assets,129);audit.prepare(4,sandbox.state[sandbox.current],sandbox.pool->activeSlots(),sandbox.pool->freeSlots(),sandbox.pool->counts(),uint32_t(sandbox.materials.entries.size()));
        require(audit.request(6,2),"Granular application audit request failed");
        require(sandbox.totalDiagnostics->request(6,sandbox.granular->contactTotals(),2),"Contact totals diagnostic request failed");
        require(sandbox.occupancyDiagnostics->request(6,sandbox.granular->occupancyStatus(),2),"Occupancy application diagnostic request failed");
        require(sandbox.neighborhoodDiagnostics->request(6,sandbox.granular->neighborhoodStatus(),2),"Neighborhood application diagnostic request failed");
        require(sandbox.granularDiagnostics->request(6,sandbox.granular->status(),2),"Granular application diagnostic request failed");
        while(audit.pending() || sandbox.granularDiagnostics->pending() || sandbox.neighborhoodDiagnostics->pending() || sandbox.occupancyDiagnostics->pending() || sandbox.totalDiagnostics->pending()){auto frame=bgfx::frame();audit.poll(frame);sandbox.granularDiagnostics->poll(frame);sandbox.neighborhoodDiagnostics->poll(frame);sandbox.occupancyDiagnostics->poll(frame);sandbox.totalDiagnostics->poll(frame);}
        require(sandbox.totalDiagnostics->latest().valid && sandbox.totalDiagnostics->latest().tick==2 && sandbox.totalDiagnostics->latest().active>=64 && sandbox.totalDiagnostics->latest().free==0,"Live contact totals missing");
        require(sandbox.occupancyDiagnostics->latest().valid && sandbox.occupancyDiagnostics->latest().tick==2 && sandbox.occupancyDiagnostics->latest().active==0 && sandbox.occupancyDiagnostics->latest().reserved==alive,"Occupancy app snapshot mismatch");
        require(sandbox.neighborhoodDiagnostics->latest().valid && sandbox.neighborhoodDiagnostics->latest().tick==2 && sandbox.neighborhoodDiagnostics->latest().rejected==20,"Neighborhood application snapshot missing");
        require(audit.latest().active==129 && !audit.latest().free && !audit.latest().reserved && !audit.latest().rejected,"Granular application ownership/state audit failed");
        require(!sandbox.granularDiagnostics->latest().active && sandbox.granularDiagnostics->latest().free>0,"Granular application cap/fault diagnostics failed");
        sandbox.queueBrush(pw::CommandType::RemoveBrush,0);sandbox.tick(1);auto empty=snapshot(sandbox);
        for(const auto& p:empty)require(!(p.identity[1]&1),"Granular application removal failed");
        sandbox.scene.count=1;sandbox.scene.minimum={0,0.0625f,0};sandbox.scene.maximum=sandbox.scene.minimum;sandbox.reset(1);bgfx::frame();
        for(unsigned i=0;i<30;++i){sandbox.tick(1);bgfx::frame();}
        auto readSettling=[&]{
            audit.prepare(4,sandbox.state[sandbox.current],sandbox.pool->activeSlots(),sandbox.pool->freeSlots(),sandbox.pool->counts(),uint32_t(sandbox.materials.entries.size()));
            require(audit.request(6,sandbox.commands.nextTick()),"Settling app audit request failed");
            require(sandbox.settlingDiagnostics->request(6,sandbox.settling->status(),sandbox.commands.nextTick()),"Settling app diagnostics request failed");
            require(sandbox.diagnostics->request(6,sandbox.pool->counts(),sandbox.commands.nextTick()),"Settling app pool request failed");
            while(audit.pending() || sandbox.settlingDiagnostics->pending() || sandbox.diagnostics->pending()) {
                auto frame=bgfx::frame();audit.poll(frame);sandbox.settlingDiagnostics->poll(frame);sandbox.diagnostics->poll(frame);
            }
            require(audit.latest().active==129 && !audit.latest().free && !audit.latest().reserved && !audit.latest().rejected,"Settling changed active/free ownership");
            require(!sandbox.settlingDiagnostics->latest().active,"Settling app tracker error");
        };
        readSettling();
        require(sandbox.settlingDiagnostics->latest().free==1 && sandbox.diagnostics->latest().active==1 && sandbox.diagnostics->latest().free==128,"Settling candidate was not retained as active matter");
        sandbox.queueBrush(pw::CommandType::ApplyImpulse,6);sandbox.tick(1);readSettling();
        require(!sandbox.settlingDiagnostics->latest().free && sandbox.diagnostics->latest().active==1,"Impulse failed to release active settling candidate");
        sandbox.reset(1);bgfx::frame();sandbox.tick(1);readSettling();
        require(!sandbox.settlingDiagnostics->latest().free && sandbox.settlingDiagnostics->latest().reserved==1,"Scene reset reused candidate history");
        sandbox.reset(0);bgfx::frame();sandbox.tick(1);readSettling();
        require(!sandbox.settlingDiagnostics->latest().free && !sandbox.settlingDiagnostics->latest().reserved && !sandbox.diagnostics->latest().active,"Empty scene retained settling candidates");
        sandbox.scene.obstacle={{0,4,0},0.9375f};sandbox.scene.minimum={0,5.00001f,0};sandbox.scene.maximum=sandbox.scene.minimum;
        sandbox.scene.count=1;sandbox.setGranular(true);bgfx::frame();
        for(unsigned i=0;i<60;++i){sandbox.tick(1);bgfx::frame();}
        readSettling();require(sandbox.terrain && sandbox.settlingDiagnostics->latest().free==1,"Live sphere failed to acquire settling support");
        const auto sphereState=snapshot(sandbox);require(std::abs(sphereState[0].positionRadius[1]-5)<5e-6,"Live sphere resting pose mismatch");
        const auto materialInstances=snapshot(sandbox,sandbox.instances,2);
        const auto historyBefore=snapshot(sandbox,sandbox.settling->history(),1);
        const auto tickBefore=sandbox.commands.nextTick();
        sandbox.debugMode=int(pw::ParticleDebugMode::ContactNormal);sandbox.prepareDebug(5);
        const auto normalInstances=snapshot(sandbox,sandbox.instances,2);
        pw::RenderInstance normalInstance;std::memcpy(&normalInstance,&normalInstances[0],sizeof(normalInstance));
        require(normalInstance.visual[0]==0xff80ff80u,"Live sphere normal did not render +Y color");
        require(std::memcmp(&normalInstances[0],&materialInstances[0],16)==0 &&
            std::memcmp(reinterpret_cast<const char*>(&normalInstances[0])+20,reinterpret_cast<const char*>(&materialInstances[0])+20,12)==0,"Normal color changed render pose/identity");
        sandbox.debugMode=0;sandbox.prepareDebug(5);
        const auto restored=snapshot(sandbox,sandbox.instances,2);
        const auto afterDebug=snapshot(sandbox),historyAfter=snapshot(sandbox,sandbox.settling->history(),1);
        for(unsigned i=0;i<sandbox.particleCount;++i){
            require(std::memcmp(&restored[i],&materialInstances[i],32)==0,"Returning to material failed while paused");
            require(std::memcmp(&historyBefore[i],&historyAfter[i],16)==0,"Normal view advanced settling history");
        }
        require(std::memcmp(sphereState.data(),afterDebug.data(),sphereState.size()*sizeof(pw::ParticleGpu))==0 && sandbox.commands.nextTick()==tickBefore,"Normal view changed simulation or tick");
        sandbox.debugMode=5;sandbox.reset(0);sandbox.prepareDebug(5);
        const auto emptyNormal=snapshot(sandbox,sandbox.instances,2);const pw::RenderInstance emptyInstance{};
        for(const auto& record:emptyNormal)require(std::memcmp(&record,&emptyInstance,32)==0,"Empty reset retained normal instances");
        sandbox.reset(1);sandbox.prepareDebug(5);const auto resetNormal=snapshot(sandbox,sandbox.instances,2);
        std::memcpy(&normalInstance,&resetNormal[0],sizeof(normalInstance));
        require(normalInstance.visual[0]==0xff80ff80u,"Paused sphere reset retained stale normals");
        pw::BrushBox sphereBrush;sphereBrush.minimum={-0.1f,3.9f,-0.1f};sphereBrush.maximum={0.1f,4.1f,0.1f};
        sandbox.queueBrush(pw::CommandType::SpawnBrush,0,1,0,&sphereBrush);
        require(sandbox.commandMessage.find("intersects sphere")!=std::string::npos,"Sphere-intersecting spawn was accepted");
        bool rejectedMode=false;try{sandbox.setGranular(false);}catch(const std::runtime_error&){rejectedMode=true;}
        require(rejectedMode && sandbox.granular && sandbox.terrain,"Sphere allowed unsupported ballistic mode");
        require(sandbox.loadPreset(assets/"scenes/empty.json") && !sandbox.terrain,"Loading non-terrain scene retained sphere renderer");
        sandbox.setGranular(false);sandbox.reset(0);bgfx::frame();
        require(!sandbox.settling && !sandbox.settlingDiagnostics && !sandbox.contactNormals,"Ballistic mode retained settling resources");
        std::cout<<"GPU live contact-normal view passed: sphere color, paused restore, immutable state/history, reset and teardown\n";
        std::cout<<"GPU live settling passed: active-list membership, impulse release, same-generation scene reset, empty scene and mode teardown\n";
        std::cout<<"GPU interactive granular path passed: radius, spawn, impulse caps, state ownership, removal and mode reset\n";
    }
}

int runBenchmark(pw::Window& window,Particles& particles,Particles* cosmetics,const std::filesystem::path& assets,const pw::RunOptions& options) {
    constexpr int width=1920,height=1080;
    int windowWidth,windowHeight;window.drawable(windowWidth,windowHeight);
    struct Target {
        bgfx::TextureHandle color=BGFX_INVALID_HANDLE,depth=BGFX_INVALID_HANDLE,readback=BGFX_INVALID_HANDLE;
        bgfx::FrameBufferHandle frame=BGFX_INVALID_HANDLE;
        ~Target() {
            bgfx::setViewFrameBuffer(10,BGFX_INVALID_HANDLE);
            if(bgfx::isValid(frame)) bgfx::destroy(frame);
            if(bgfx::isValid(readback)) bgfx::destroy(readback);
            if(bgfx::isValid(depth)) bgfx::destroy(depth);
            if(bgfx::isValid(color)) bgfx::destroy(color);
        }
    } target;
    target.color=bgfx::createTexture2D(width,height,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_RT);
    target.depth=bgfx::createTexture2D(width,height,false,1,bgfx::TextureFormat::D32F,BGFX_TEXTURE_RT);
    if(!bgfx::isValid(target.color) || !bgfx::isValid(target.depth)) throw std::runtime_error("Benchmark render target unavailable");
    const bgfx::TextureHandle attachments[]{target.color,target.depth};
    target.frame=bgfx::createFrameBuffer(2,attachments,false);
    if(!bgfx::isValid(target.frame)) throw std::runtime_error("Benchmark framebuffer unavailable");
    bgfx::setViewFrameBuffer(10,target.frame);
    const auto* caps=bgfx::getCaps();
    const bool indirect=options.indirect && particles.visibility->supportsIndirect();
    pw::Benchmark benchmark(options);
    float view[16],projection[16],viewProjection[16];
    bx::mtxLookAt(view,{std::sin(.65f)*24,12,std::cos(.65f)*24},{0,3,0},{0,1,0},bx::Handedness::Right);
    bx::mtxProj(projection,55,float(width)/height,.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
    bx::mtxMul(viewProjection,view,projection);
    const auto frustum=pw::extractFrustum(viewProjection,caps->homogeneousDepth);
    bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
    bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
    bgfx::setViewTransform(10,view,projection);
    // Finish the seed upload before warmup; one fixed tick per benchmark frame.
    bgfx::frame();
    const uint64_t total=uint64_t(options.warmup)+options.frames;
    std::chrono::steady_clock::time_point measuredBegin;
    for(uint64_t frame=0;frame<total;++frame) {
        if(frame==options.warmup) {
            particles.diagnostics->request(6,particles.pool->counts(),frame);
            while(particles.diagnostics->pending()) {particles.diagnostics->poll(bgfx::frame());SDL_PumpEvents();}
            measuredBegin=std::chrono::steady_clock::now();
        }
        const auto begin=std::chrono::steady_clock::now();
        SDL_Event event;
        while(SDL_PollEvent(&event)) if(event.type==SDL_EVENT_QUIT || event.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED)
            throw std::runtime_error("Benchmark cancelled");
        int w,h;window.drawable(w,h);
        if(w!=windowWidth || h!=windowHeight || (SDL_GetWindowFlags(window.get())&SDL_WINDOW_MINIMIZED))
            throw std::runtime_error("Benchmark interrupted by resize/minimize");
        particles.tick(1);if(cosmetics) cosmetics->tick(1);
        particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),frustum);
        if(cosmetics) cosmetics->visibility->prepare(5,cosmetics->instances,cosmetics->pool->activeSlots(),cosmetics->pool->counts(),frustum);
        bgfx::touch(10);particles.render(10,indirect);if(cosmetics) cosmetics->render(10,indirect);
        const auto submitted=bgfx::frame();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(frame>=options.warmup) {benchmark.add(submitted,frame+1,ms);benchmark.observeMemory(*bgfx::getStats());}
        benchmark.collect(*bgfx::getStats());
    }
    // Bounded drain for delayed GPU timestamps; excluded from measured samples.
    for(unsigned i=0;i<32 && benchmark.gpuSamples()<options.frames;++i) {
        bgfx::frame();benchmark.collect(*bgfx::getStats());SDL_PumpEvents();
    }
    particles.diagnostics->request(6,particles.pool->counts(),total);
    particles.visibleDiagnostics->request(6,particles.visibility->counts(),total);
    if(cosmetics) {
        cosmetics->diagnostics->request(6,cosmetics->pool->counts(),total);
        cosmetics->visibleDiagnostics->request(6,cosmetics->visibility->counts(),total);
    }
    while(particles.diagnostics->pending() || particles.visibleDiagnostics->pending() ||
          (cosmetics && (cosmetics->diagnostics->pending() || cosmetics->visibleDiagnostics->pending()))) {
        const auto frame=bgfx::frame();particles.diagnostics->poll(frame);particles.visibleDiagnostics->poll(frame);
        if(cosmetics) {cosmetics->diagnostics->poll(frame);cosmetics->visibleDiagnostics->poll(frame);}SDL_PumpEvents();
    }
    benchmark.setCompletedWall(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-measuredBegin).count());
    target.readback=bgfx::createTexture2D(width,height,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    if(!bgfx::isValid(target.readback)) throw std::runtime_error("Benchmark image validation allocation failed");
    std::vector<uint32_t> pixels(width*height);
    bgfx::TextureRegion dst;dst.handle=target.readback;bgfx::TextureRegion src;src.handle=target.color;
    bgfx::blit(7,dst,src);const auto imageReady=bgfx::read(dst,pixels.data());
    while(bgfx::frame()<imageReady) SDL_PumpEvents();
    const auto& counts=particles.diagnostics->latest();
    const auto& visible=particles.visibleDiagnostics->latest();
    if(!counts.valid || !visible.valid || counts.active!=options.capacity || counts.free!=0 || counts.rejected ||
       counts.tick!=total || visible.tick!=total || !visible.active || visible.active>counts.active)
        throw std::runtime_error("Benchmark final GPU counts failed validation");
    const auto differentPixels=uint32_t(std::count_if(pixels.begin(),pixels.end(),[&](auto pixel){return pixel!=pixels.front();}));
    if(!differentPixels) throw std::runtime_error("Benchmark rendered a uniform image instead of particles");
    uint32_t cosmeticAlive=0,cosmeticVisible=0,cosmeticPixels=0,physicalPixels=differentPixels;
    if(cosmetics) {
        const auto& c=cosmetics->diagnostics->latest();const auto& v=cosmetics->visibleDiagnostics->latest();
        if(!c.valid || !v.valid || c.tick!=total || v.tick!=total || c.active!=options.cosmeticCapacity ||
           c.free || c.rejected || !v.active || v.active>c.active)
            throw std::runtime_error("Benchmark cosmetic GPU counts failed validation");
        cosmeticAlive=c.active;cosmeticVisible=v.active;
        // Verify each draw independently outside the measured window.
        auto isolatedPixels=[&](Particles& engine) {
            bgfx::touch(10);engine.render(10,indirect);bgfx::frame();
            bgfx::blit(7,dst,src);const auto ready=bgfx::read(dst,pixels.data());
            while(bgfx::frame()<ready) SDL_PumpEvents();
            return uint32_t(std::count_if(pixels.begin(),pixels.end(),[&](auto pixel){return pixel!=pixels.front();}));
        };
        physicalPixels=isolatedPixels(particles);cosmeticPixels=isolatedPixels(*cosmetics);
        if(!physicalPixels || !cosmeticPixels) throw std::runtime_error("Benchmark isolated pool draw produced a uniform image");
    }
    benchmark.write(assets,width,height,indirect,counts.active,visible.active,differentPixels,*caps,cosmeticAlive,cosmeticVisible,cosmeticPixels,physicalPixels);
    return 0;
}

int runReplay(pw::Window& window,Particles& particles,const std::filesystem::path& assets,const pw::RunOptions& options) {
    const auto& recording=particles.session.recording();
    if(!options.report.empty()) {
        const auto input=std::filesystem::weakly_canonical(options.replay);
        const auto output=std::filesystem::weakly_canonical(options.report);
        if(input==output || (std::filesystem::exists(output) && std::filesystem::equivalent(input,output)))
            throw std::runtime_error("Replay report must not overwrite the recording");
    }
    pw::ParticleAudit audit(assets,particles.particleCount);
    const auto* caps=bgfx::getCaps();
    const bool indirect=options.indirect && particles.visibility->supportsIndirect();
    int width,height;window.drawable(width,height);
    float view[16],projection[16],vp[16];
    bx::mtxLookAt(view,{std::sin(.65f)*24,12,std::cos(.65f)*24},{0,3,0},{0,1,0},bx::Handedness::Right);
    bx::mtxProj(projection,55,float(width)/height,.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
    bx::mtxMul(vp,view,projection);const auto frustum=pw::extractFrustum(vp,caps->homogeneousDepth);
    bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
    bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
    bgfx::setViewTransform(10,view,projection);
    bgfx::frame();
    pw::ReportedMemoryPeaks memoryPeaks;memoryPeaks.observe(pw::reportedMemory(*bgfx::getStats()));
    std::string failure;
    auto nextProgress=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    auto inspect=[&] {
        audit.prepare(4,particles.state[particles.current],particles.pool->activeSlots(),particles.pool->freeSlots(),
                      particles.pool->counts(),uint32_t(particles.materials.entries.size()));
    };
    while(particles.session.replaying() && failure.empty()) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) if(event.type==SDL_EVENT_QUIT || event.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED) failure="Replay cancelled";
        int w,h;window.drawable(w,h);
        if(w!=width || h!=height || (SDL_GetWindowFlags(window.get())&SDL_WINDOW_MINIMIZED)) failure="Replay interrupted by resize/minimize";
        if(!failure.empty()) break;
        particles.tick(1);inspect();
        particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),frustum);
        bgfx::touch(10);particles.render(10,indirect);bgfx::frame();
        memoryPeaks.observe(pw::reportedMemory(*bgfx::getStats()));
        if(std::chrono::steady_clock::now()>=nextProgress) {
            std::cout<<"Replay tick "<<particles.session.completedTicks()<<"/"<<recording.ticks<<std::endl;
            nextProgress=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        }
    }
    // Include the initial state for zero-tick recordings and cancelled runs.
    inspect();
    const auto ticks=particles.session.completedTicks();
    audit.request(6,ticks);particles.diagnostics->request(6,particles.pool->counts(),ticks);
    while(audit.pending() || particles.diagnostics->pending()) {
        const auto frame=bgfx::frame();memoryPeaks.observe(pw::reportedMemory(*bgfx::getStats()));audit.poll(frame);particles.diagnostics->poll(frame);SDL_PumpEvents();
    }
    const auto& a=audit.latest();const auto& pool=particles.diagnostics->latest();
    if(!a.valid || !pool.valid || a.tick!=ticks || pool.tick!=ticks || a.active!=particles.particleCount ||
       a.free || a.reserved || a.rejected || uint64_t(pool.active)+pool.free!=particles.particleCount)
        failure="GPU ownership, numeric state, identity or pool count audit failed";
    if(failure.empty() && (!particles.session.finished() || ticks!=recording.ticks)) failure="Replay did not finish";
    std::ifstream buildFile(assets/"build.json");nlohmann::json build;buildFile>>build;
    const auto applied=std::count_if(recording.commands.begin(),recording.commands.end(),[&](const auto& command){return command.tick<ticks;});
    nlohmann::json report={{"schema_version",1},{"passed",failure.empty()},{"failure",failure},
        {"recording",std::filesystem::absolute(options.replay).string()},{"software_renderer_requested",options.softwareGpu},
        {"input",nlohmann::json::parse(pw::encodeRecording(recording,particles.materials))},{"capacity",particles.particleCount},
        {"requested_ticks",recording.ticks},{"completed_ticks",ticks},{"applied_commands",applied},
        {"resources",pw::resourceReport(pw::resourceUsage(particles.particleCount,uint32_t(particles.materials.entries.size()),particles.visibility->supportsIndirect(),true),*bgfx::getStats())},
        {"reported_memory_peaks",pw::resourcePeakReport(memoryPeaks,"replay_initial_submission_and_final_readback_snapshots")},
        {"recorded_workgroup_size",recording.workgroupSize},{"workgroup_size",pw::workgroupSize},
        {"materials_hash",recording.materialsHash},{"audit_every_tick",true},{"build",build},
        {"backend",bgfx::getRendererName(caps->rendererType)},{"indirect",indirect},
        {"final_snapshot",{{"checked_slots",a.active},{"ownership_error",bool(a.free)},{"state_error",bool(a.reserved)},
            {"identity_error",bool(a.rejected)},{"active",pool.active},{"free",pool.free},{"rejected_spawns",pool.rejected}}},
        {"note","Replays configuration and input; does not assert identical particle trajectories across GPUs."}};
    if(!options.report.empty()) {
        const std::filesystem::path path=options.report;
        if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path);output<<report.dump(2)<<'\n';output.close();
        if(!output) throw std::runtime_error("Cannot write replay report");
    }
    std::cout<<"Replay "<<(failure.empty()?"passed":"failed")<<": "<<ticks<<" ticks, "<<applied<<" commands, "<<pool.active<<" active particles"<<std::endl;
    return failure.empty()?0:1;
}

int runSoak(pw::Window& window,Particles& particles,const std::filesystem::path& assets,const pw::RunOptions& options) {
    using Json=nlohmann::json;
    pw::ParticleAudit audit(assets,options.capacity);
    const auto* caps=bgfx::getCaps();
    const bool indirect=options.indirect && particles.visibility->supportsIndirect();
    int width,height;window.drawable(width,height);
    float view[16],projection[16],vp[16];
    bx::mtxLookAt(view,{std::sin(.65f)*24,12,std::cos(.65f)*24},{0,3,0},{0,1,0},bx::Handedness::Right);
    bx::mtxProj(projection,55,float(width)/height,.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
    bx::mtxMul(vp,view,projection);const auto frustum=pw::extractFrustum(vp,caps->homogeneousDepth);
    bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
    bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
    bgfx::setViewTransform(10,view,projection);
    bgfx::frame();
    std::ifstream buildFile(assets/"build.json");Json build;buildFile>>build;
    std::ifstream materialsFile(assets/"materials"/"default.json");Json materials;materialsFile>>materials;
    Json shaderFingerprints=Json::object();
    for(const auto& entry:std::filesystem::directory_iterator(assets/"shaders")) if(entry.path().extension()==".json") {
        std::ifstream input(entry.path());Json manifest;input>>manifest;
        shaderFingerprints[entry.path().filename().string()]=manifest.at("fingerprint");
    }
    const auto begin=std::chrono::steady_clock::now();
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
    pw::ReportedMemoryPeaks memoryPeaks;
    uint64_t ticks=0,rejectedCommands=0,spawnRequests=0,discardedSamples=0;
    uint64_t lastSample=UINT64_MAX;
    double nextSample=0,nextProgress=0;
    bool sawFull=false,sawEmpty=false,sawOverflow=false,sawSettling=false;
    std::string failure;
    Json samples=Json::array();
    auto collect=[&](uint32_t frame) {
        memoryPeaks.observe(pw::reportedMemory(*bgfx::getStats()));
        audit.poll(frame);particles.diagnostics->poll(frame);
        if(particles.granularDiagnostics) {
            for(auto* ring:{particles.granularDiagnostics.get(),particles.settlingDiagnostics.get(),particles.neighborhoodDiagnostics.get(),particles.occupancyDiagnostics.get(),particles.totalDiagnostics.get()}) ring->poll(frame);
            const auto& g=particles.granularDiagnostics->latest();const auto& h=particles.settlingDiagnostics->latest();
            if((g.valid && g.active) || (h.valid && h.active)) failure="Latched granular solver or settling fault";
        }
        const auto& a=audit.latest();const auto& pool=particles.diagnostics->latest();
        if(!a.valid || !pool.valid || a.tick!=pool.tick || a.sequence==lastSample) return;
        if(particles.granularDiagnostics && (!particles.granularDiagnostics->latest().valid || !particles.settlingDiagnostics->latest().valid ||
            particles.granularDiagnostics->latest().tick!=a.tick || particles.settlingDiagnostics->latest().tick!=a.tick)) return;
        if(particles.granularDiagnostics) for(auto* ring:{particles.neighborhoodDiagnostics.get(),particles.occupancyDiagnostics.get(),particles.totalDiagnostics.get()})
            if(!ring->latest().valid || ring->latest().tick!=a.tick) return;
        lastSample=a.sequence;
        const bool valid=a.active==options.capacity && !a.free && !a.reserved && !a.rejected && uint64_t(pool.active)+pool.free==options.capacity;
        if(!valid) failure="GPU ownership, numeric state or identity audit failed";
        sawFull=sawFull || pool.active==options.capacity;sawEmpty=sawEmpty || pool.active==0;sawOverflow=sawOverflow || pool.rejected>0;
        if(samples.size()<8192) samples.push_back({{"tick",a.tick},{"elapsed_seconds",elapsed()},{"checked_slots",a.active},
            {"ownership_error",bool(a.free)},{"state_error",bool(a.reserved)},{"identity_error",bool(a.rejected)},
            {"active",pool.active},{"free",pool.free},{"rejected_spawns",pool.rejected}});
        else ++discardedSamples;
        if(particles.granularDiagnostics) {
            const auto& g=particles.granularDiagnostics->latest();const auto& h=particles.settlingDiagnostics->latest();
            const auto& n=particles.neighborhoodDiagnostics->latest();const auto& o=particles.occupancyDiagnostics->latest();
            const auto& t=particles.totalDiagnostics->latest();
            const uint64_t candidates=uint64_t(t.active)+(uint64_t(t.free)<<32);
            const uint64_t contacts=uint64_t(t.reserved)+(uint64_t(t.rejected)<<32);
            // Four position iterations and two response candidate walks per substep.
            const uint64_t candidateBound=uint64_t(particles.granular->plan().substeps)*6*options.capacity*options.capacity;
            if(n.active || o.active) failure="Granular neighborhood or occupancy diagnostic fault";
            if(n.free>options.capacity || n.rejected!=particles.granular->plan().substeps*5)
                failure="Granular neighborhood diagnostic bounds failed";
            if(o.free>o.reserved || o.reserved>options.capacity || o.rejected>o.reserved ||
                ((o.reserved==0)!=(o.free==0)) || ((o.reserved==0)!=(o.rejected==0)))
                failure="Granular occupancy diagnostic bounds failed";
            if(contacts>candidates || candidates>candidateBound)
                failure="Granular contact totals bounds failed";
            if(samples.size() && samples.back()["tick"]==a.tick) {
                samples.back()["neighborhood"]={{"error",n.active},{"maximum_candidates",n.free},{"maximum_probes",n.reserved},{"contact_solves",n.rejected}};
                samples.back()["occupancy"]={{"error",o.active},{"occupied_cells",o.free},{"indexed_solids",o.reserved},{"maximum_cell_population",o.rejected}};
                samples.back()["contact_totals"]={{"candidate_visits",candidates},{"directed_contacts",contacts}};
            }
            sawSettling|=h.free>0;
            if(uint64_t(h.free)+h.reserved>pool.active || h.rejected>h.free) failure="Settling counts exceed active matter";
            if(samples.size() && samples.back()["tick"]==a.tick) samples.back()["granular"]={{"solver_error",g.active},{"speed_caps",g.free},{"maximum_contact_degree",g.reserved},{"boundary_events",g.rejected},
                {"tracking_error",h.active},{"settling_candidates",h.free},{"acquiring",h.reserved},{"contact_grace",h.rejected}};
        }
    };
    auto request=[&] {
        const bool accepted=audit.request(6,ticks);
        if(particles.diagnostics->request(6,particles.pool->counts(),ticks)!=accepted)
            throw std::runtime_error("Soak snapshot rings lost synchronization");
        if(particles.granularDiagnostics) {
            const bool g=particles.granularDiagnostics->request(6,particles.granular->status(),ticks);
            const bool h=particles.settlingDiagnostics->request(6,particles.settling->status(),ticks);
            const bool n=particles.neighborhoodDiagnostics->request(6,particles.granular->neighborhoodStatus(),ticks);
            const bool o=particles.occupancyDiagnostics->request(6,particles.granular->occupancyStatus(),ticks);
            const bool t=particles.totalDiagnostics->request(6,particles.granular->contactTotals(),ticks);
            if(g!=accepted || h!=accepted || n!=accepted || o!=accepted || t!=accepted) throw std::runtime_error("Granular soak snapshot rings lost synchronization");
        }
    };
    auto pending=[&]{return audit.pending() || particles.diagnostics->pending() ||
        (particles.granularDiagnostics && (particles.granularDiagnostics->pending() || particles.settlingDiagnostics->pending() || particles.neighborhoodDiagnostics->pending() || particles.occupancyDiagnostics->pending() || particles.totalDiagnostics->pending()));};
    pw::CommandLimits limits;limits.materialCount=uint32_t(particles.materials.entries.size());
    auto enqueue=[&](pw::Command command) {
        command.tick=particles.commands.nextTick();
        if(auto reason=particles.commands.enqueue(command,limits);!reason.empty()) throw std::runtime_error("Soak command unexpectedly rejected: "+std::string(reason));
        if(command.type==pw::CommandType::SpawnBrush) spawnRequests+=command.count;
    };
    while(elapsed()<options.soakSeconds && failure.empty()) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) if(event.type==SDL_EVENT_QUIT || event.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED) failure="Soak cancelled";
        int w,h;window.drawable(w,h);
        if(w!=width || h!=height || (SDL_GetWindowFlags(window.get())&SDL_WINDOW_MINIMIZED)) failure="Soak interrupted by resize/minimize";
        if(!failure.empty()) break;
        const auto phase=ticks%256;
        pw::Command spawn;spawn.type=pw::CommandType::SpawnBrush;spawn.count=4096;
        spawn.material=particles.liveMaterialIds[(ticks/8)%particles.liveMaterialIds.size()];
        spawn.minimum=options.scene.empty()?std::array<float,3>{-6,1,-6}:particles.scene.minimum;
        spawn.maximum=options.scene.empty()?std::array<float,3>{6,11,6}:particles.scene.maximum;
        if(phase==0) {
            enqueue(spawn);
            auto invalid=spawn;invalid.tick=particles.commands.nextTick();invalid.count=4097;
            if(particles.commands.enqueue(invalid,limits).empty()) throw std::runtime_error("Invalid soak command was accepted");
            ++rejectedCommands;
        }
        if(phase==1 || phase==131) for(auto material:particles.liveMaterialIds) {
            pw::Command remove;remove.type=pw::CommandType::RemoveBrush;remove.material=material;
            remove.minimum={-8,-1,-8};remove.maximum={8,1024,8};enqueue(remove);enqueue(remove);
        }
        if((phase>=2 && phase<=130) || (phase>=132 && phase<=199)) {
            spawn.parameters[3]=phase>=132?.25f:0.0f;enqueue(spawn);enqueue(spawn);
        }
        if(phase>=132 && phase<=199 && phase%17==0) {
            auto impulse=spawn;impulse.type=pw::CommandType::ApplyImpulse;impulse.count=0;impulse.parameters={1,2,-1,0};enqueue(impulse);
        }
        particles.tick(1);++ticks;
        audit.prepare(4,particles.state[particles.current],particles.pool->activeSlots(),particles.pool->freeSlots(),particles.pool->counts(),uint32_t(particles.materials.entries.size()),particles.scene.obstacle);
        particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),frustum);
        if(elapsed()>=nextSample || phase==0 || phase==1 || phase==129 || phase==130 || phase==131 || phase==215) {
            request();
            if(elapsed()>=nextSample) nextSample=elapsed()+1;
        }
        bgfx::touch(10);particles.render(10,indirect);collect(bgfx::frame());
        if(elapsed()>=nextProgress) {
            std::cout<<"Soak "<<int(elapsed())<<"/"<<options.soakSeconds<<" seconds; tick "<<ticks<<"; audited snapshots "<<samples.size()<<std::endl;
            nextProgress=elapsed()+30;
        }
    }
    while(pending()) collect(bgfx::frame());
    request();
    while(pending()) collect(bgfx::frame());
    if(particles.commands.rejected()!=rejectedCommands) failure="Unexpected CPU rejection count";
    if(failure.empty() && (ticks<256 || !sawFull || !sawEmpty || !sawOverflow)) failure="Soak did not complete required full/empty/overflow coverage";
    if(failure.empty() && particles.granular && !sawSettling) failure="Granular soak did not observe settling candidates";
    const auto& finalAudit=audit.latest();const auto& finalPool=particles.diagnostics->latest();
    if(failure.empty() && (!finalAudit.valid || !finalPool.valid || finalAudit.tick!=ticks || finalPool.tick!=ticks ||
        (particles.granularDiagnostics && (particles.granularDiagnostics->latest().tick!=ticks || particles.settlingDiagnostics->latest().tick!=ticks || particles.neighborhoodDiagnostics->latest().tick!=ticks || particles.occupancyDiagnostics->latest().tick!=ticks || particles.totalDiagnostics->latest().tick!=ticks))))
        failure="Soak final snapshots do not match the final tick";
    auto resources=pw::resourceUsage(particles.particleCount,uint32_t(particles.materials.entries.size()),particles.visibility->supportsIndirect(),true);
    if(particles.granular){resources.granular=pw::granularPayloadBytes(particles.particleCount)+pw::settlingPayloadBytes(particles.particleCount);resources.diagnosticTextures+=320;resources.diagnosticHost+=240;}
    if(particles.terrain)resources.terrainGeometry=pw::SphereTerrain::payloadBytes;
    Json report={{"schema_version",1},{"workload",particles.granular?(options.scene.empty()?"granular_256_tick_pool_cycle_v2":"granular_scene_256_tick_pool_cycle_v1"):"bounded_256_tick_pool_cycle_v1"},{"software_renderer_requested",options.softwareGpu},{"seed",particles.scene.seed},{"tick_seconds",1.0/60.0},{"workgroup_size",pw::workgroupSize},{"audit_every_tick",true},{"passed",failure.empty()},{"failure",failure},{"capacity",options.capacity},
        {"resources",pw::resourceReport(resources,*bgfx::getStats())},
        {"reported_memory_peaks",pw::resourcePeakReport(memoryPeaks,"soak_submission_and_final_readback_snapshots")},
        {"requested_seconds",options.soakSeconds},{"elapsed_seconds",elapsed()},{"ticks",ticks},
        {"slot_checks",ticks*options.capacity},{"completed_cycles",ticks/256},{"spawn_requests",spawnRequests},
        {"intentional_cpu_rejections",rejectedCommands},{"saw_full",sawFull},{"saw_empty",sawEmpty},{"saw_overflow",sawOverflow},
        {"machine",{{"platform",SDL_GetPlatform()},{"logical_cpus",SDL_GetNumLogicalCPUCores()},{"ram_mb",SDL_GetSystemRAM()},{"gpu_vendor_id",caps->vendorId},{"gpu_device_id",caps->deviceId}}},
        {"final_snapshot",{{"tick",finalAudit.tick},{"checked_slots",finalAudit.active},{"ownership_error",bool(finalAudit.free)},{"state_error",bool(finalAudit.reserved)},{"identity_error",bool(finalAudit.rejected)},{"active",finalPool.active},{"free",finalPool.free},{"rejected_spawns",finalPool.rejected}}},
        {"backend",bgfx::getRendererName(caps->rendererType)},{"resolution",{{"width",width},{"height",height}}},
        {"indirect",indirect},{"vsync",false},{"build",build},{"shader_fingerprints",shaderFingerprints},{"materials",materials},{"samples",samples},{"discarded_samples",discardedSamples},
        {"notes",Json::array({"One 1/60-second tick per frame; duration is wall-clock time.","Every tick audits every slot; error flags remain latched across readback intervals.","256-tick cycles cover fill/overflow, duplicate removal, reuse, short lifetimes and impulses.","This is a correctness soak, not a performance benchmark or reference-machine certification."})}};
    report["scene"]=Json::parse(pw::encodeScene(particles.scene));
    report["spawn_bounds"]={{"minimum",options.scene.empty()?std::array<float,3>{-6,1,-6}:particles.scene.minimum},
        {"maximum",options.scene.empty()?std::array<float,3>{6,11,6}:particles.scene.maximum}};
    report["sphere_containment_audited_every_tick"]=particles.scene.obstacle.radius>0;
    if(particles.granular) {
        const auto& g=particles.granularDiagnostics->latest();const auto& h=particles.settlingDiagnostics->latest();
        report["granular"]={{"position_friction","static_traction_approach_v3"},{"radius",0.0625},{"speed_limit",12},{"substeps",4},{"iterations",4},{"saw_settling_candidates",sawSettling},
            {"solver_error",g.active},{"tracking_error",h.active},{"final_candidates",h.free},{"final_acquiring",h.reserved},{"final_contact_grace",h.rejected},
            {"initial_scene",{{"name",particles.scene.name},{"count",particles.scene.count},{"minimum",particles.scene.minimum},{"maximum",particles.scene.maximum},{"materials",particles.scene.materials}}}};
        if(particles.scene.obstacle.radius>0)report["notes"].push_back("Every live slot independently checks sphere separation with 1e-6 m float allowance; scene spawn bounds remain outside the expanded collider.");
        report["notes"].push_back("All five granular diagnostic rings are sampled at the ownership-audit tick; occupancy describes the last position-solve input. Contact totals count repeated directed visits, not unique pairs.");
        report["notes"].push_back("Granular solver and settling faults latch every tick; synchronized samples check candidates remain bounded by active matter.");
    }
    if(!failure.empty() && particles.granular) {
        // These are the final submitted tick's buffers, not necessarily the first
        // fault tick (diagnostic readbacks are asynchronous). Keep raw words so
        // signed zero, nonfinite floats and identity lanes survive JSON exactly.
        Json captured={{"tick",ticks},{"maximum_slots",4096},{"capacity",particles.particleCount},
            {"scope","final submitted tick; may follow the first fault; first 4096 slots"},
            {"layout","four vec4 lanes per slot: position/radius, velocity/inverse mass, thermal, identity; uint32 bit patterns"}};
        try {
            const auto save=[&](const char* name,bgfx::DynamicVertexBufferHandle buffer) {
                const auto state=snapshotParticles(particles,assets,buffer);
                Json rows=Json::array();
                for(const auto& particle:state) {
                    std::array<uint32_t,16> words{};
                    static_assert(sizeof(words)==sizeof(particle));
                    std::memcpy(words.data(),&particle,sizeof(particle));rows.push_back(words);
                }
                captured[name]=std::move(rows);
            };
            save("committed",particles.state[particles.current]);
            const auto stages=particles.granular->diagnosticStates();
            constexpr std::array names{"force","sweep","position","reconstruction","response","finish"};
            for(unsigned i=0;i<stages.size();++i)save(names[i],stages[i]);
        } catch(const std::exception& e) {captured["capture_error"]=e.what();}
        report["failure_state"]=std::move(captured);
    }
    std::filesystem::path path=options.report;if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);output<<report.dump(2)<<'\n';output.close();if(!output) throw std::runtime_error("Cannot write soak report");
    std::cout<<"Soak "<<(failure.empty()?"passed":"failed")<<": "<<options.report<<std::endl;
    return failure.empty()?0:1;
}

}
int main(int argc, char** argv) {
    try {
        const auto options=pw::parseOptions(argc,argv);
        const auto frameLimit=options.frames;
        const bool gpuTestOnly=options.gpuTest,allowIndirect=options.indirect;
        const bool benchmarking=!options.benchmark.empty();
        pw::Window window;
        Graphics graphics(window,!benchmarking && !options.soakSeconds && !options.granularCheck && options.replay.empty() && !options.softwareGpu,options.softwareGpu);
        const auto* caps=bgfx::getCaps();
        auto capabilities=pw::capabilityReport(*caps,allowIndirect);
        capabilities["backend"]=bgfx::getRendererName(caps->rendererType);
        capabilities["software_renderer_requested"]=options.softwareGpu;
        if(options.softwareGpu) std::cout<<"Execution mode: explicitly requested D3D11 WARP software validation; not hardware certification\n";
        std::cout << "Backend: " << capabilities["backend"] << "; capability status: " << capabilities["status"] << '\n';
        for(const auto& check:capabilities["checks"])
            if(check["status"]!="supported") std::cout << check["feature"] << ": " << check["action"] << '\n';
        if(!options.capabilities.empty()) {
            const std::filesystem::path reportPath=options.capabilities;
            if(reportPath.has_parent_path()) std::filesystem::create_directories(reportPath.parent_path());
            std::ofstream report(reportPath);report << capabilities.dump(2) << '\n';report.close();
            if(!report) throw std::runtime_error("Cannot write capability report: "+reportPath.string());
            return capabilities["can_run"].get<bool>()?0:1;
        }
        if(!capabilities["can_run"].get<bool>()) throw std::runtime_error("Required GPU capabilities unavailable; see individual failures above");
        std::cout << "Submission: compact visible instances; workgroup: " << pw::workgroupSize << '\n';
        const char* basePath=SDL_GetBasePath();
        if(!basePath) throw std::runtime_error("SDL cannot resolve application assets");
        const std::filesystem::path assets=std::filesystem::path(basePath)/"assets";
        pw::runGpuLayoutTest(assets);
        pw::runGpuPoolTest(assets);
        pw::runGpuVisibilityTest(assets);
        pw::runGpuAuditTest(assets);
        if(options.granularCheck) {
            auto report=pw::runGpuGranularStability(assets);report["software_renderer_requested"]=options.softwareGpu;
            std::ifstream manifest(assets/"build.json");if(manifest)manifest>>report["build"];
            const std::filesystem::path destination=options.report;if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
            std::ofstream file(destination);file<<report.dump(2)<<'\n';file.close();if(!file)throw std::runtime_error("Cannot write granular stability report");
            return report["passed"].get<bool>()?0:1;
        }
        if(gpuTestOnly) {pw::runGpuVoxelWorldTest(assets);pw::runGpuVoxelSweepTest(assets);pw::runGpuPrefixScanTest(assets);pw::runGpuRadixSortTest(assets);pw::runGpuSpatialKeysTest(assets);pw::runGpuCellRangesTest(assets);pw::runGpuCellHashTest(assets);pw::runGpuNeighborsTest(assets);pw::runGpuJacobiContactsTest(assets);pw::runGpuVelocityReconstructionTest(assets);pw::runGpuContactVelocityTest(assets);pw::runGpuSweptBoundsTest(assets);pw::runGpuGranularStepTest(assets);pw::runGpuSettlingTest(assets);pw::runGpuParticleDebugTest(assets);pw::runGpuContactNormalsTest(assets);pw::runImGuiRendererTest(assets);pw::runShaderReloadTest(assets);runGpuPoolIsolationTest(assets);return 0;}
        std::optional<pw::CommandRecording> replay;
        if(!options.replay.empty()) replay=pw::loadRecording(options.replay,pw::loadMaterials(assets/"materials/default.json"));
        pw::ScenePreset granularScene;granularScene.name="Granular sand drop";granularScene.count=std::min(256u,options.capacity);
        granularScene.minimum={-0.75f,1,-0.75f};granularScene.maximum={0.75f,2.5f,0.75f};granularScene.materials={"Sand"};
        Particles particles(assets,replay?replay->capacity:options.capacity,options.scene,replay?&replay->scene:(options.granular?&granularScene:nullptr));
        if(options.granular || particles.scene.obstacle.radius>0) particles.setGranular(true);
        if(replay) {particles.beginReplay(std::move(*replay),false);return runReplay(window,particles,assets,options);}
        if(options.soakSeconds) return runSoak(window,particles,assets,options);
        std::unique_ptr<Particles> cosmetics;
        if(options.cosmeticCapacity) {
            pw::ScenePreset sparks; sparks.name=benchmarking?"Benchmark cosmetic cloud":"Cosmetic sparks";sparks.count=benchmarking?options.cosmeticCapacity:0;sparks.materials={"Wood"};
            if(benchmarking) sparks.seed=4242;
            cosmetics=std::make_unique<Particles>(assets,options.cosmeticCapacity,std::filesystem::path{},&sparks);
        }
        if(benchmarking) return runBenchmark(window,particles,cosmetics.get(),assets,options);
        int cosmeticBurst=1024;float cosmeticLifetime=2,cosmeticLift=6;
        GuiContext gui(window.get());
        pw::ImGuiRenderer guiRenderer(assets);

        char recordingPath[512]{};
        if(char* preferences=SDL_GetPrefPath("ParticleWorld","ParticleWorld")) {
            const auto path=(std::filesystem::path(preferences)/"recording.json").string();
            std::snprintf(recordingPath,sizeof(recordingPath),"%s",path.c_str());SDL_free(preferences);
        }
        std::string recordingMessage;
        std::string shaderMessage;
        bool reloadShaders=false;
        pw::ReportedMemoryPeaks memoryPeaks;
        pw::FixedStep clock;
        bool running=true,paused=false,cull=true;
        bool indirect=allowIndirect && particles.visibility->supportsIndirect();
        std::cout<<"Drawing: "<<(indirect?"GPU indirect":"masked fixed-capacity")<<'\n';
        uint64_t ticks=0; uint32_t frames=0, completedFrame=0; double lag=0;
        int width=0,height=0;
        float yaw=0.65f, distance=options.granular?8.0f:24.0f, elevation=options.granular?5.0f:12.0f, impulseStrength=6, spawnLifetime=10;
        const float orbitTarget=options.granular?1.0f:3.0f;
        int presetSelection=0;
        constexpr const char* presetFiles[]{"ballistic.json","empty.json","sand_drop.json","sphere_drop.json"};
        int spawnCount=1024,pointerTool=0;
        uint32_t brushMaterial=0;
        float brushHeight=.5f,brushHalfSize=1;
        std::optional<ImVec2> brushPointer;
        pw::FlyCamera flyCamera;
        bool freeCamera=false,cameraCaptured=false;
        float cameraSpeed=8;
        std::string cameraError;
        auto releaseCamera=[&] {
            if(cameraCaptured) SDL_SetWindowRelativeMouseMode(window.get(),false);
            cameraCaptured=false;
        };
        auto previous=std::chrono::steady_clock::now();
        while(running) {
            if(particles.granularDiagnostics) {
                particles.granularDiagnostics->poll(completedFrame);
                particles.neighborhoodDiagnostics->poll(completedFrame);
                particles.occupancyDiagnostics->poll(completedFrame);
                particles.totalDiagnostics->poll(completedFrame);
                const auto& fault=particles.granularDiagnostics->latest();
                if(fault.valid && fault.active) {particles.granularFault=true;paused=true;clock.reset();}
            }
            if(particles.settlingDiagnostics) {
                particles.settlingDiagnostics->poll(completedFrame);
                const auto& tracking=particles.settlingDiagnostics->latest();
                if(tracking.valid && tracking.active) {particles.granularFault=true;paused=true;clock.reset();}
            }
            particles.diagnostics->poll(completedFrame);
            particles.visibleDiagnostics->poll(completedFrame);
            if(cosmetics) {cosmetics->diagnostics->poll(completedFrame);cosmetics->visibleDiagnostics->poll(completedFrame);}
            SDL_Event event;
            float lookX=0,lookY=0;
            std::optional<ImVec2> brushClick;
            std::array<bool,SDL_SCANCODE_COUNT> keyPressed{};
            bool cancelCamera=false;
            while(SDL_PollEvent(&event)) {
                // A button event carries its own position. Preserve it before queuing
                // the button, including synthetic clicks without a preceding motion event.
                if((event.type==SDL_EVENT_MOUSE_BUTTON_DOWN || event.type==SDL_EVENT_MOUSE_BUTTON_UP) &&
                    event.button.windowID==SDL_GetWindowID(window.get()))
                    ImGui::GetIO().AddMousePosEvent(event.button.x,event.button.y);
                if(event.type==SDL_EVENT_WINDOW_FOCUS_LOST || event.type==SDL_EVENT_WINDOW_MINIMIZED ||
                   (event.type==SDL_EVENT_KEY_DOWN && event.key.scancode==SDL_SCANCODE_ESCAPE) ||
                   (event.type==SDL_EVENT_MOUSE_BUTTON_UP && event.button.button==SDL_BUTTON_RIGHT)) {
                    releaseCamera();cancelCamera=true;lookX=lookY=0;
                }
                if(event.type==SDL_EVENT_MOUSE_MOTION && event.motion.windowID==SDL_GetWindowID(window.get()))
                    brushPointer=ImVec2(event.motion.x,event.motion.y);
                if((event.type==SDL_EVENT_MOUSE_BUTTON_DOWN || event.type==SDL_EVENT_MOUSE_BUTTON_UP) &&
                   event.button.windowID==SDL_GetWindowID(window.get())) brushPointer=ImVec2(event.button.x,event.button.y);
                if(event.type==SDL_EVENT_MOUSE_MOTION && cameraCaptured) {lookX+=event.motion.xrel;lookY+=event.motion.yrel;}
                if(event.type==SDL_EVENT_KEY_DOWN && !event.key.repeat &&
                   event.key.scancode>=0 && event.key.scancode<SDL_SCANCODE_COUNT)
                    keyPressed[event.key.scancode]=true;
                if(event.type==SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button==SDL_BUTTON_LEFT &&
                   event.button.windowID==SDL_GetWindowID(window.get())) brushClick=ImVec2(event.button.x,event.button.y);
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
                releaseCamera();clock.reset(); SDL_Delay(16); continue;
            }
            if(w!=width || h!=height) { width=w;height=h; bgfx::SwapChain swap; swap.nwh=window.nativeHandle(); swap.width=uint32_t(w); swap.height=uint32_t(h); bgfx::reset(BGFX_RESET_VSYNC,&swap); }
            ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame();
            if(std::getenv("PW_TRACE_INPUT") && (ImGui::IsMouseClicked(0) || ImGui::IsMouseReleased(0)))
                std::cerr << "ImGui mouse " << ImGui::GetIO().MousePos.x << "," << ImGui::GetIO().MousePos.y << " clicked=" << ImGui::IsMouseClicked(0) << " released=" << ImGui::IsMouseReleased(0) << '\n';
            ImGui::SetNextWindowPos(ImVec2(16,16),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(460,680),ImGuiCond_FirstUseEver);
            ImGui::Begin("Particle World - foundation");
            ImGui::Text("Backend: %s",bgfx::getRendererName(caps->rendererType));
            ImGui::Text("Capacity: %u | Seed: %u",particles.particleCount,particles.scene.seed);
            const auto& poolStats=particles.diagnostics->latest();
            if(poolStats.valid) {
                ImGui::Text("Active: %u | Free: %u",poolStats.active,poolStats.free);
                ImGui::Text("Rejected spawns: %u",poolStats.rejected);
                ImGui::TextDisabled("GPU snapshot at tick %llu",static_cast<unsigned long long>(poolStats.tick));
            } else {
                ImGui::TextDisabled("Active: ... | Free: ...");
                ImGui::TextDisabled("Rejected spawns: ...");
                ImGui::TextDisabled("Waiting for GPU pool snapshot...");
            }
            const auto& visibleStats=particles.visibleDiagnostics->latest();
            if(visibleStats.valid) ImGui::Text("Visible (delayed): %u",visibleStats.active);
            else ImGui::TextDisabled("Waiting for visibility snapshot...");
            ImGui::Text("Tick: %llu",static_cast<unsigned long long>(ticks));
            bool collisions=bool(particles.granular);
            ImGui::BeginDisabled(particles.session.active());
            if(ImGui::Checkbox("Particle collisions (resets scene)",&collisions)) {
                try {particles.setGranular(collisions);clock.reset();ticks=0;lag=0;reloadShaders=false;}
                catch(const std::exception& error) {particles.solverMessage=error.what();}
            }
            ImGui::EndDisabled();
            if(particles.granular) {
                ImGui::Text("Granular: 12 m/s limit | %u substeps | 4 iterations",particles.granular->plan().substeps);
                const auto& g=particles.granularDiagnostics->latest();
                if(g.valid) ImGui::Text("Caps: %u | Contacts: %u | Boundary events: %u",g.free,g.reserved,g.rejected);
                else ImGui::TextDisabled("Waiting for solver snapshot...");
                const auto& occupancy=particles.occupancyDiagnostics->latest();
                if(occupancy.valid) {
                    if(occupancy.active)ImGui::Text("Cell occupancy unavailable: error 0x%x",occupancy.active);
                    else {
                        ImGui::Text("Cell snapshot tick %llu: %u occupied",static_cast<unsigned long long>(occupancy.tick),occupancy.free);
                        ImGui::Text("Solids: %u | Max/cell: %u | Mean/cell: %.2f",occupancy.reserved,occupancy.rejected,occupancy.free?double(occupancy.reserved)/occupancy.free:0.0);
                        ImGui::TextDisabled("Last position-solve input; no cell population cap");
                    }
                }
                const auto& neighbors=particles.neighborhoodDiagnostics->latest();
                if(neighbors.valid) {
                    ImGui::Text("Neighborhood tick %llu",static_cast<unsigned long long>(neighbors.tick));
                    ImGui::Text("Max candidates/query: %u | Max probes/query: %u",neighbors.free,neighbors.reserved);
                    ImGui::TextDisabled("%u contact solves; candidates include self; no neighbor cap",neighbors.rejected);
                    if(neighbors.active)ImGui::Text("Neighborhood error mask: 0x%x (partial metrics)",neighbors.active);
                }
                const auto& total=particles.totalDiagnostics->latest();
                if(total.valid && g.valid && total.tick==g.tick) {
                    const auto candidates=std::uint64_t(total.active)+(std::uint64_t(total.free)<<32);
                    const auto contacts=std::uint64_t(total.reserved)+(std::uint64_t(total.rejected)<<32);
                    ImGui::Text("Total candidate visits: %llu",static_cast<unsigned long long>(candidates));
                    ImGui::Text("Directed contacts: %llu",static_cast<unsigned long long>(contacts));
                    ImGui::TextDisabled(g.active?"Failed tick: successful-phase totals only":"Per tick; repeated passes, not unique pairs");
                }
                const auto& tracking=particles.settlingDiagnostics->latest();
                if(tracking.valid) ImGui::Text("Settling candidates: %u | Acquiring: %u | Contact grace: %u",tracking.free,tracking.reserved,tracking.rejected);
                else ImGui::TextDisabled("Waiting for settling snapshot...");
                ImGui::TextDisabled("Candidates remain active particles");
                ImGui::Combo("Particle view",&particles.debugMode,"Material\0Speed\0Spatial cells\0Settling\0Velocity direction\0Contact normal\0");
                if(particles.debugMode==1) ImGui::TextDisabled("Speed: blue = 0, red = 12+ m/s");
                if(particles.debugMode==2) ImGui::TextDisabled("Color by 0.125 m cell; magenta = outside grid");
                if(particles.debugMode==3) ImGui::TextWrapped("Gray: active | Yellow: acquiring | Green: candidate | Orange: contact grace | Magenta: tracking fault");
                if(particles.debugMode==4) ImGui::TextWrapped("Direction: RGB = XYZ, dark = negative, bright = positive. Gray: zero | Magenta: invalid velocity");
                if(particles.debugMode==5) ImGui::TextWrapped("Closest contact normal: RGB = XYZ, dark = negative, bright = positive. Gray: no contact | Magenta: diagnostic/solver fault. Rebuilds contacts while paused.");
                if(particles.granularFault) ImGui::TextWrapped("Solver error 0x%x / tracking error 0x%x: paused. Reset or load a valid scene to resume.",g.active,tracking.active);
            }
            if(!particles.solverMessage.empty()) ImGui::TextWrapped("%s",particles.solverMessage.c_str());
            ImGui::Checkbox("Paused",&paused); ImGui::SameLine();
            bool step=ImGui::Button("Single tick"); if(step) paused=true;
            ImGui::BeginDisabled(particles.session.active());
            if(ImGui::Button("Reset scene")) {particles.reset();clock.reset();ticks=0;lag=0;}
            ImGui::SameLine();
            if(ImGui::Button("Empty scene")) {particles.reset(0);clock.reset();ticks=0;lag=0;}
            if(ImGui::CollapsingHeader("Scene presets")) {
                ImGui::TextWrapped("Current: %s",particles.scene.name.c_str());
                ImGui::Combo("Preset",&presetSelection,"Ballistic cloud\0Empty workspace\0Sand drop\0Sand over sphere\0");
                if(ImGui::Button("Load preset") && particles.loadPreset(assets/"scenes"/presetFiles[presetSelection])) {clock.reset();ticks=0;lag=0;}
                if(!particles.scenePath.empty()) {
                    ImGui::SameLine();
                    if(ImGui::Button("Reload scene file") && particles.loadPreset(particles.scenePath)) {clock.reset();ticks=0;lag=0;}
                }
                if(!particles.sceneMessage.empty()) ImGui::TextWrapped("%s",particles.sceneMessage.c_str());
            }
            const char* sceneLabel=particles.sceneMaterial<0?"Preset materials":particles.materials.entries[particles.sceneMaterial].name.c_str();
            if(ImGui::BeginCombo("Scene material",sceneLabel)) {
                if(ImGui::Selectable("Preset materials",particles.sceneMaterial<0)) {particles.sceneMaterial=-1;particles.reset();clock.reset();ticks=0;}
                for(std::uint32_t id=0;id<particles.materials.entries.size();++id) {
                    const auto& material=particles.materials.entries[id];
                    if(material.gpu.identity[1]!=0) continue;
                    if(ImGui::Selectable(material.name.c_str(),particles.sceneMaterial==int(id))) {particles.sceneMaterial=int(id);particles.reset();clock.reset();ticks=0;}
                }
                ImGui::EndCombo();
            }
            ImGui::BeginDisabled(bool(particles.granular));
            if(ImGui::Button("Reload shaders")) reloadShaders=true;
            ImGui::EndDisabled();
            if(!shaderMessage.empty()) ImGui::TextWrapped("%s",shaderMessage.c_str());
            if(ImGui::Button("Reload materials")) particles.reloadPending=true;
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Reloads %s",particles.materialPath.string().c_str());
            if(particles.reloadPending) ImGui::TextWrapped("Reload queued for next tick (step when paused).");
            if(!particles.materialMessage.empty()) ImGui::TextWrapped("%s",particles.materialMessage.c_str());
            ImGui::EndDisabled();
            ImGui::Text("Profiles: %u",unsigned(particles.materials.entries.size()));
            ImGui::SliderFloat("Lift speed (m/s)",&impulseStrength,0,20);
            if(ImGui::Button("Lift center particles")) particles.queueBrush(pw::CommandType::ApplyImpulse,impulseStrength);
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Adds upward velocity in the center 8 x 12 x 8 m box. Targets the selected scene material or the preset's first material.");
            if(ImGui::Button("Remove center particles")) particles.queueBrush(pw::CommandType::RemoveBrush,0);
            ImGui::SliderInt("Spawn count",&spawnCount,1,4096);
            ImGui::SliderFloat("Lifetime (s)",&spawnLifetime,0,30);
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Zero keeps particles alive until removed; positive values expire after that many simulation seconds.");
            if(ImGui::Button("Spawn particles")) particles.queueBrush(pw::CommandType::SpawnBrush,0,uint32_t(spawnCount),spawnLifetime);
            ImGui::Text("Queued: %u | Rejected commands: %llu",unsigned(particles.commands.pending()),static_cast<unsigned long long>(particles.commands.rejected()));
            if(!particles.commandMessage.empty()) ImGui::TextWrapped("%s",particles.commandMessage.c_str());
            if(brushMaterial>=particles.materials.entries.size() || particles.materials.entries[brushMaterial].gpu.identity[1]!=0)
                brushMaterial=particles.liveMaterialIds.front();
            if(ImGui::CollapsingHeader("Pointer brush")) {
                ImGui::Combo("Pointer tool",&pointerTool,"Off\0Spawn\0Remove\0Lift\0");
                if(ImGui::BeginCombo("Brush material",particles.materials.entries[brushMaterial].name.c_str())) {
                    for(uint32_t id=0;id<particles.materials.entries.size();++id) {
                        const auto& material=particles.materials.entries[id];
                        if(material.gpu.identity[1]==0 && ImGui::Selectable(material.name.c_str(),brushMaterial==id)) brushMaterial=id;
                    }
                    ImGui::EndCombo();
                }
                ImGui::SliderFloat("Brush half-size (m)",&brushHalfSize,.1f,3);
                ImGui::SliderFloat("Brush plane Y (m)",&brushHeight,0,12);
                ImGui::TextWrapped("Left click the scene once per brush. Uses spawn count/lifetime or lift speed above; step to apply while paused. The wire box shows the affected region on the chosen horizontal plane.");
            }
            if(ImGui::CollapsingHeader("Cosmetic sparks")) {
                if(cosmetics) {
                    const auto& counts=cosmetics->diagnostics->latest();
                    ImGui::Text("Independent capacity: %u",cosmetics->particleCount);
                    if(counts.valid) ImGui::Text("Alive: %u | Free: %u | Dropped: %u",counts.active,counts.free,counts.rejected);
                    ImGui::Text("Queued: %u | Rejected commands: %llu",unsigned(cosmetics->commands.pending()),static_cast<unsigned long long>(cosmetics->commands.rejected()));
                    ImGui::BeginDisabled(particles.session.active());
                    ImGui::SliderInt("Sparks per burst",&cosmeticBurst,1,4096,"%d",ImGuiSliderFlags_AlwaysClamp);
                    ImGui::SliderFloat("Spark lifetime (s)",&cosmeticLifetime,.1f,10,"%.3f",ImGuiSliderFlags_AlwaysClamp);
                    ImGui::SliderFloat("Spark lift (m/s)",&cosmeticLift,0,20,"%.3f",ImGuiSliderFlags_AlwaysClamp);
                    if(ImGui::Button("Emit cosmetic sparks")) cosmetics->queueBrush(pw::CommandType::SpawnBrush,cosmeticLift,uint32_t(cosmeticBurst),cosmeticLifetime);
                    ImGui::SameLine();if(ImGui::Button("Clear sparks")) cosmetics->reset(0);
                    ImGui::EndDisabled();
                    if(!cosmetics->commandMessage.empty()) ImGui::TextWrapped("%s",cosmetics->commandMessage.c_str());
                    ImGui::TextWrapped("Visual effects only. Overflow drops sparks without consuming physical slots. Recording/replay clears sparks and disables emission.");
                } else ImGui::TextWrapped("Disabled by --cosmetic-capacity 0.");
            }
            ImGui::Checkbox("Frustum culling",&cull);
            if(particles.visibility->supportsIndirect()) {ImGui::SameLine();ImGui::Checkbox("Indirect draw",&indirect);}
            if(ImGui::Checkbox("Free camera",&freeCamera)) {
                releaseCamera();
                if(freeCamera) flyCamera.fromOrbit(yaw,distance,elevation,orbitTarget);
            }
            ImGui::SameLine();
            if(ImGui::Button("Reset view")) {yaw=.65f;distance=options.granular?8.0f:24.0f;elevation=options.granular?5.0f:12.0f;flyCamera.fromOrbit(yaw,distance,elevation,orbitTarget);releaseCamera();}
            if(freeCamera) {
                ImGui::SliderFloat("Camera speed (m/s)",&cameraSpeed,1,25);
                ImGui::TextWrapped("Click the scene: WASD move, Q/E down/up, Shift faster. Hold right mouse to look; Esc releases it.");
                const auto& eye=flyCamera.position();ImGui::Text("Camera: %.2f, %.2f, %.2f m",eye[0],eye[1],eye[2]);
            } else {
                ImGui::SliderFloat("Orbit",&yaw,-3.14f,3.14f);
                ImGui::SliderFloat("Distance",&distance,8,45);
                ImGui::SliderFloat("Elevation",&elevation,2,25);
            }
            if(!cameraError.empty()) ImGui::TextWrapped("%s",cameraError.c_str());
            const auto* stats=bgfx::getStats();
            if(stats->cpuTimerFreq) ImGui::Text("CPU frame: %.2f ms",1000.0*double(stats->cpuTimeFrame)/double(stats->cpuTimerFreq));
            if(stats->gpuTimerFreq) ImGui::Text("GPU frame: %.2f ms",1000.0*double(stats->gpuTimeEnd-stats->gpuTimeBegin)/double(stats->gpuTimerFreq));
            ImGui::Text("Dropped wall time: %.3f s",lag);
            ImGui::TextWrapped(particles.granular?"Granular sphere contacts and bounded walls. Spawns are clipped to the enclosure.":"Ballistic pool: spawn, remove, lifetime and box contacts.");
            const auto panelPosition=ImGui::GetWindowPos(),panelSize=ImGui::GetWindowSize();
            ImGui::End();
            ImGui::SetNextWindowPos(ImVec2(492,16),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(420,210),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowCollapsed(true,ImGuiCond_FirstUseEver);
            ImGui::Begin("Recording");
            if(particles.granular) ImGui::TextWrapped("Recording currently supports the ballistic solver. Switch off particle collisions to record or replay.");
            ImGui::BeginDisabled(bool(particles.granular));
            ImGui::InputText("File",recordingPath,sizeof(recordingPath));
            ImGui::BeginDisabled(particles.session.active());
            if(ImGui::Button("Record preset from reset")) {
                particles.captureFromReset();if(cosmetics) cosmetics->reset(0);reloadShaders=false;clock.reset();ticks=0;lag=0;recordingMessage.clear();
            }
            ImGui::EndDisabled();
            if(particles.session.active() && ImGui::Button("Stop session")) {particles.session.stop();paused=true;}
            ImGui::BeginDisabled(particles.session.active() || !particles.session.hasRecording());
            if(ImGui::Button("Save recording")) {
                try {pw::saveRecording(recordingPath,particles.session.recording(),particles.materials);recordingMessage="Recording saved";}
                catch(const std::exception& error) {recordingMessage=error.what();}
            }
            ImGui::SameLine();
            if(ImGui::Button("Replay in memory")) {
                bool valid=false;
                try {pw::validateRecording(particles.session.recording(),particles.materials);valid=true;}
                catch(const std::exception& error) {recordingMessage=error.what();}
                if(valid) {
                    particles.beginReplay(particles.session.recording());if(cosmetics) cosmetics->reset(0);clock.reset();ticks=0;lag=0;paused=true;recordingMessage.clear();
                }
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(particles.session.active());
            if(ImGui::Button("Load replay file")) {
                std::optional<pw::CommandRecording> candidate;
                try {
                    candidate=pw::loadRecording(recordingPath,particles.materials);
                    if(candidate->capacity!=particles.particleCount) throw std::runtime_error("Capacity differs; use --replay to allocate the recorded capacity");
                } catch(const std::exception& error) {recordingMessage=error.what();candidate.reset();}
                if(candidate) {particles.beginReplay(std::move(*candidate));if(cosmetics) cosmetics->reset(0);clock.reset();ticks=0;lag=0;paused=true;recordingMessage.clear();}
            }
            ImGui::EndDisabled();
            if(particles.session.hasRecording()) ImGui::Text("%llu ticks | %u applied commands",static_cast<unsigned long long>(particles.session.recording().ticks),unsigned(particles.session.recording().commands.size()));
            ImGui::TextWrapped("%s",particles.session.message().c_str());
            if(!recordingMessage.empty()) ImGui::TextWrapped("%s",recordingMessage.c_str());
            ImGui::EndDisabled();
            ImGui::End();
            auto resources=pw::resourceUsage(particles.particleCount,uint32_t(particles.materials.entries.size()),particles.visibility->supportsIndirect());
            if(particles.terrain)resources.terrainGeometry=pw::SphereTerrain::payloadBytes;
            if(cosmetics) resources+=pw::resourceUsage(cosmetics->particleCount,uint32_t(cosmetics->materials.entries.size()),cosmetics->visibility->supportsIndirect());
            if(particles.contactNormals)resources.contactNormals=pw::contactNormalPayloadBytes(particles.particleCount);
            if(particles.granular) {resources.granular=pw::granularPayloadBytes(particles.particleCount)+pw::settlingPayloadBytes(particles.particleCount);resources.diagnosticTextures+=320;resources.diagnosticHost+=240;}
            memoryPeaks.observe(pw::reportedMemory(*stats));
            pw::drawResourceUsage(resources,*stats,memoryPeaks);
            const auto& input=ImGui::GetIO();
            const bool focused=(SDL_GetWindowFlags(window.get())&SDL_WINDOW_INPUT_FOCUS)!=0;
            if(freeCamera && focused && !cancelCamera) {
                if(!cameraCaptured && !input.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    cameraCaptured=SDL_SetWindowRelativeMouseMode(window.get(),true);
                    cameraError=cameraCaptured?"":std::string("Cannot capture mouse: ")+SDL_GetError();
                }
                if(cameraCaptured) flyCamera.look(lookX,lookY);
                if(!input.WantCaptureKeyboard && (!input.WantCaptureMouse || cameraCaptured) &&
                   (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) || cameraCaptured)) {
                    const auto* keys=SDL_GetKeyboardState(nullptr);
                    // Preserve a tap whose down/up events both arrived before this poll.
                    auto down=[&](SDL_Scancode key){return keys[key] || keyPressed[key];};
                    flyCamera.move(float(down(SDL_SCANCODE_W))-float(down(SDL_SCANCODE_S)),
                        float(down(SDL_SCANCODE_D))-float(down(SDL_SCANCODE_A)),
                        float(down(SDL_SCANCODE_E))-float(down(SDL_SCANCODE_Q)),float(elapsed),
                        cameraSpeed*((down(SDL_SCANCODE_LSHIFT) || down(SDL_SCANCODE_RSHIFT))?3.0f:1.0f));
                }
            } else releaseCamera();
            if(reloadShaders) {
                reloadShaders=false;
                try {
                    pw::ShaderReload reload(assets);
                    particles.stageShaders(reload);if(cosmetics) cosmetics->stageShaders(reload);guiRenderer.stageShaders(reload);reload.commit();
                    shaderMessage="Shaders reloaded; scene preserved.";
                } catch(const std::exception& error) {shaderMessage=std::string("Reload rejected; working shaders retained: ")+error.what();}
            }
            float view[16],projection[16];
            if(freeCamera) {
                const auto& eye=flyCamera.position();const auto target=flyCamera.target();
                bx::mtxLookAt(view,{eye[0],eye[1],eye[2]},{target[0],target[1],target[2]},{0,1,0},bx::Handedness::Right);
            } else bx::mtxLookAt(view,{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance},{0,orbitTarget,0},{0,1,0},bx::Handedness::Right);
            bx::mtxProj(projection,55,float(width)/float(height),0.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
            float viewProjection[16];bx::mtxMul(viewProjection,view,projection);
            if(pointerTool && focused && !cameraCaptured && !cancelCamera) {
                int logicalWidth,logicalHeight;SDL_GetWindowSize(window.get(),&logicalWidth,&logicalHeight);
                const pw::BrushPoint eye=freeCamera?flyCamera.position():pw::BrushPoint{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance};
                const pw::BrushPoint target=freeCamera?flyCamera.target():pw::BrushPoint{0,3,0};
                const auto cursor=brushClick.value_or(brushPointer.value_or(input.MousePos));
                const bool overPanel=cursor.x>=panelPosition.x && cursor.y>=panelPosition.y &&
                    cursor.x<panelPosition.x+panelSize.x && cursor.y<panelPosition.y+panelSize.y;
                if(!overPanel && !ImGui::IsPopupOpen(nullptr,ImGuiPopupFlags_AnyPopup) && logicalWidth>0 && logicalHeight>0) {
                    const auto ray=pw::pointerRay(eye,target,cursor.x/logicalWidth,cursor.y/logicalHeight,float(width)/height);
                    const auto box=ray?pw::planeBrush(*ray,brushHeight,brushHalfSize):std::nullopt;
                    if(box) {
                        std::array<std::optional<std::array<float,2>>,8> corners;
                        for(unsigned corner=0;corner<8;++corner) {
                            pw::BrushPoint point;
                            for(unsigned axis=0;axis<3;++axis) point[axis]=(corner&(1u<<axis))?box->maximum[axis]:box->minimum[axis];
                            corners[corner]=pw::projectBrushPoint(point,viewProjection,float(logicalWidth),float(logicalHeight));
                        }
                        const ImU32 color=pointerTool==2?IM_COL32(255,100,100,255):IM_COL32(80,240,190,255);
                        for(unsigned corner=0;corner<8;++corner) for(unsigned axis=0;axis<3;++axis) {
                            const unsigned other=corner^(1u<<axis);
                            if(corner<other && corners[corner] && corners[other])
                                ImGui::GetBackgroundDrawList()->AddLine({(*corners[corner])[0],(*corners[corner])[1]},
                                    {(*corners[other])[0],(*corners[other])[1]},color,2);
                        }
                        if(brushClick && !input.WantCaptureMouse && !ImGui::IsAnyItemActive()) {
                            const auto type=pointerTool==1?pw::CommandType::SpawnBrush:pointerTool==2?pw::CommandType::RemoveBrush:pw::CommandType::ApplyImpulse;
                            particles.queueBrush(type,type==pw::CommandType::ApplyImpulse?impulseStrength:0,
                                type==pw::CommandType::SpawnBrush?uint32_t(spawnCount):0,spawnLifetime,&*box,brushMaterial);
                        }
                    } else if(brushClick && !input.WantCaptureMouse) particles.commandMessage="Brush missed the selected plane inside the world bounds";
                }
            }
            if(particles.granularFault) {paused=true;step=false;}
            auto batch=clock.advance(elapsed,paused,step); lag+=batch.droppedSeconds;
            for(unsigned i=0;i<batch.count;++i) {
                const bool sessionWasActive=particles.session.active();
                particles.tick(uint16_t(i+1));if(cosmetics && !sessionWasActive) cosmetics->tick(uint16_t(i+1));++ticks;
                if(sessionWasActive && !particles.session.active()) {paused=true;clock.reset();break;}
            }
            if(particles.granularDiagnostics && ticks && (frames%8==0 || step || !particles.granularDiagnostics->latest().valid))
                particles.granularDiagnostics->request(6,particles.granular->status(),ticks);
            if(particles.occupancyDiagnostics && ticks && (frames%8==0 || step || !particles.occupancyDiagnostics->latest().valid))
                particles.occupancyDiagnostics->request(6,particles.granular->occupancyStatus(),ticks);
            if(particles.neighborhoodDiagnostics && ticks && (frames%8==0 || step || !particles.neighborhoodDiagnostics->latest().valid))
                particles.neighborhoodDiagnostics->request(6,particles.granular->neighborhoodStatus(),ticks);
            if(particles.totalDiagnostics && ticks && (frames%8==0 || step || !particles.totalDiagnostics->latest().valid))
                particles.totalDiagnostics->request(6,particles.granular->contactTotals(),ticks);
            if(particles.settlingDiagnostics && ticks && (frames%8==0 || step || !particles.settlingDiagnostics->latest().valid))
                particles.settlingDiagnostics->request(6,particles.settling->status(),ticks);
            if(frames%8==0 || step || !particles.diagnostics->latest().valid)
                particles.diagnostics->request(6,particles.pool->counts(),ticks);
            if(frames%8==0 || step || !particles.visibleDiagnostics->latest().valid)
                particles.visibleDiagnostics->request(6,particles.visibility->counts(),ticks);
            particles.prepareDebug(5);
            particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),pw::extractFrustum(viewProjection,caps->homogeneousDepth),cull);
            if(cosmetics) {
                cosmetics->visibility->prepare(5,cosmetics->instances,cosmetics->pool->activeSlots(),cosmetics->pool->counts(),pw::extractFrustum(viewProjection,caps->homogeneousDepth),cull);
                if(frames%8==0 || step || !cosmetics->diagnostics->latest().valid) {
                    cosmetics->diagnostics->request(6,cosmetics->pool->counts(),cosmetics->commands.nextTick());
                    cosmetics->visibleDiagnostics->request(6,cosmetics->visibility->counts(),cosmetics->commands.nextTick());
                }
            }
            bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
            bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
            bgfx::setViewTransform(10,view,projection); bgfx::touch(10);
            particles.render(10,indirect);if(cosmetics) cosmetics->render(10,indirect);
            ImGui::Render(); guiRenderer.render(ImGui::GetDrawData(),20);
            completedFrame=bgfx::frame();
            ++frames;
            if(frameLimit && frames>=frameLimit) running=false;
        }
        releaseCamera();
        if(particles.granularDiagnostics && ticks) {
            while(particles.granularDiagnostics->pending()) particles.granularDiagnostics->poll(bgfx::frame());
            particles.granularDiagnostics->request(6,particles.granular->status(),ticks);
            while(particles.granularDiagnostics->pending()) particles.granularDiagnostics->poll(bgfx::frame());
            const auto& g=particles.granularDiagnostics->latest();
            std::cout<<"Granular final status: error="<<g.active<<", caps="<<g.free<<", contacts="<<g.reserved<<", boundary_events="<<g.rejected<<"\n";
            if(g.active) throw std::runtime_error("Granular solver fault remained latched at shutdown");
        }
        if(particles.settlingDiagnostics && ticks) {
            while(particles.settlingDiagnostics->pending()) particles.settlingDiagnostics->poll(bgfx::frame());
            particles.settlingDiagnostics->request(6,particles.settling->status(),ticks);
            while(particles.settlingDiagnostics->pending()) particles.settlingDiagnostics->poll(bgfx::frame());
            const auto& s=particles.settlingDiagnostics->latest();
            std::cout<<"Settling final status: error="<<s.active<<", candidates="<<s.free<<", acquiring="<<s.reserved<<", contact_grace="<<s.rejected<<"\n";
            if(s.active) throw std::runtime_error("Settling fault remained latched at shutdown");
        }
        std::cout << "Clean shutdown after " << ticks << " ticks\n";
        return 0;
    } catch(const std::exception& error) {std::cerr << "Particle World: " << error.what() << '\n';return 1;}
}
