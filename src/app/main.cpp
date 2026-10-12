#include "platform/Window.hpp"
#include "Benchmark.hpp"
#include "core/RunOptions.hpp"
#include "core/FixedStep.hpp"
#include "core/CommandQueue.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include "render/ImGuiRenderer.hpp"
#include "render/Shaders.hpp"
#include "render/GpuLayoutTest.hpp"
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
#include <fstream>

namespace {
struct Graphics {
    Graphics(pw::Window& window,bool vsync=true) {
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
    explicit Particles(const std::filesystem::path& assets,uint32_t capacity):particleCount(capacity) {
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
            uploadMaterials(materials);
            pool=std::make_unique<pw::ParticlePool>(assets,particleCount);
            diagnostics=std::make_unique<pw::PoolDiagnostics>(assets);
            visibility=std::make_unique<pw::ParticleVisibility>(assets,particleCount);
            visibleDiagnostics=std::make_unique<pw::PoolDiagnostics>(assets);
            reset();
        } catch (...) { release(); throw; }
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
            uploadMaterials(candidate);
            materials=std::move(candidate);
            materialMessage="Materials reloaded";
        } catch(const std::exception& error) {materialMessage=std::string("Reload rejected: ")+error.what();}
    }
    void release() {
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
        diagnostics->beginEpoch();
        visibleDiagnostics->beginEpoch();
        bgfx::setViewMode(0,bgfx::ViewMode::Sequential);
        commands.reset(); commandMessage.clear();
        for (auto& h: state) { if(bgfx::isValid(h)) bgfx::destroy(h); h = BGFX_INVALID_HANDLE; }
        if(bgfx::isValid(instances)) bgfx::destroy(instances);
        instances = BGFX_INVALID_HANDLE;
        liveMaterialIds.clear();
        for(std::uint32_t id=0;id<materials.entries.size();++id)
            if(materials.entries[id].gpu.identity[1]==0 && (sceneMaterial<0 || sceneMaterial==int(id))) liveMaterialIds.push_back(id);
        if(liveMaterialIds.empty()) throw std::runtime_error("Scene requires a solid material profile");
        std::vector<pw::ParticleGpu> initial(particleCount);
        std::mt19937 rng(42);
        // Explicit integer mapping keeps authored initial positions reproducible.
        auto random = [&] { return float(rng() >> 8) / 16777216.0f; };
        for(uint32_t i=0;i<particleCount;++i) initial[i].identity[2]=i;
        for(uint32_t i=0; i<std::min(initialCount,particleCount); ++i) {
            auto& p=initial[i];
            p.positionRadius[0]=(random()-0.5f)*12;
            p.positionRadius[1]=1+random()*10;
            p.positionRadius[2]=(random()-0.5f)*12;
            p.positionRadius[3]=0.045f;
            const auto materialId=liveMaterialIds[i%liveMaterialIds.size()];
            const auto& material=materials.entries[materialId].gpu;
            const float radius=p.positionRadius[3];
            p.velocityInvMass[3]=1.0f/(4.0f/3.0f*3.14159265f*radius*radius*radius*material.mechanical[0]);
            p.thermalState[0]=293.15f;
            p.identity[0]=materialId; p.identity[1]=1; p.identity[2]=i; p.identity[3]=1;
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
        const float params[]{0,float(particleCount),float(materials.entries.size()),0};
        bgfx::setUniform(uniform,params);
        bgfx::setBuffer(0,upload,bgfx::Access::Read);
        bgfx::setBuffer(1,state[0],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer,bgfx::Access::Read);
        bgfx::dispatch(0,initialize,(particleCount+127)/128);
        bgfx::destroy(upload); // bgfx defers destruction past submitted work.
        current=0;
        pool->rebuild(0,state[current],true);
    }
    void queueBrush(pw::CommandType type,float strength,uint32_t count=0,float lifetime=0) {
        // One spatial impulse for the chosen scene material; mixed scenes use Stone.
        pw::Command command;
        command.tick=commands.nextTick(); command.type=type;
        command.minimum={-4,0,-4}; command.maximum={4,12,4};
        command.parameters={0,strength,0,lifetime}; command.count=count;
        if(type==pw::CommandType::SpawnBrush) {command.minimum={-2,6,-2};command.maximum={2,10,2};}
        command.material=sceneMaterial<0?0u:unsigned(sceneMaterial);
        pw::CommandLimits limits; limits.materialCount=uint32_t(materials.entries.size());
        const auto reason=commands.enqueue(command,limits);
        commandMessage=reason.empty()?"Brush queued for next tick":std::string(reason);
    }
    void tick(uint16_t view) {
        reloadAtTickBoundary();
        bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
        const auto batch=commands.consume();
        for(const auto& command:batch) {
            pw::CommandLimits limits;limits.materialCount=uint32_t(materials.entries.size());
            if(!pw::validate(command,limits,command.tick).empty()) {
                commandMessage="Queued command rejected after material reload";continue;
            }
            pool->apply(view,state[current],materialBuffer,uint32_t(materials.entries.size()),command);
            commandMessage="Brush submitted (spawns limited by free capacity)";
        }
        const float params[]{float(pw::FixedStep::tickSeconds),float(particleCount),float(materials.entries.size()),0};
        bgfx::setUniform(uniform, params);
        bgfx::setBuffer(0,state[current],bgfx::Access::Read);
        bgfx::setBuffer(1,state[1-current],bgfx::Access::Write);
        bgfx::setBuffer(2,instances,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer,bgfx::Access::Read);
        bgfx::dispatch(view,compute,(particleCount+127)/128);
        current=1-current;
        pool->rebuild(view,state[current]);
    }
    void render(uint16_t view,bool indirect) {
        bgfx::setVertexBuffer(0,vertices); bgfx::setIndexBuffer(indices);
        bgfx::setInstanceDataBuffer(visibility->instances(),0,particleCount);
        bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);
        if(indirect && visibility->supportsIndirect()) bgfx::submit(view,draw,visibility->arguments(),0,1);
        else bgfx::submit(view,draw);
    }
};
int runBenchmark(pw::Window& window,Particles& particles,const std::filesystem::path& assets,const pw::RunOptions& options) {
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
        particles.tick(1);
        particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),frustum);
        bgfx::touch(10);particles.render(10,indirect);
        const auto submitted=bgfx::frame();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(frame>=options.warmup) benchmark.add(submitted,frame+1,ms);
        benchmark.collect(*bgfx::getStats());
    }
    // Bounded drain for delayed GPU timestamps; excluded from measured samples.
    for(unsigned i=0;i<32 && benchmark.gpuSamples()<options.frames;++i) {
        bgfx::frame();benchmark.collect(*bgfx::getStats());SDL_PumpEvents();
    }
    particles.diagnostics->request(6,particles.pool->counts(),total);
    particles.visibleDiagnostics->request(6,particles.visibility->counts(),total);
    while(particles.diagnostics->pending() || particles.visibleDiagnostics->pending()) {
        const auto frame=bgfx::frame();particles.diagnostics->poll(frame);particles.visibleDiagnostics->poll(frame);SDL_PumpEvents();
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
    if(!counts.valid || !visible.valid || counts.active!=options.capacity || counts.free!=0 || visible.active>counts.active)
        throw std::runtime_error("Benchmark final GPU counts failed validation");
    const auto differentPixels=uint32_t(std::count_if(pixels.begin(),pixels.end(),[&](auto pixel){return pixel!=pixels.front();}));
    if(!differentPixels) throw std::runtime_error("Benchmark rendered a uniform image instead of particles");
    benchmark.write(assets,width,height,indirect,counts.active,visible.active,differentPixels,*caps);
    return 0;
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
    uint64_t ticks=0,rejectedCommands=0,spawnRequests=0,discardedSamples=0;
    uint64_t lastSample=UINT64_MAX;
    double nextSample=0,nextProgress=0;
    bool sawFull=false,sawEmpty=false,sawOverflow=false;
    std::string failure;
    Json samples=Json::array();
    auto collect=[&](uint32_t frame) {
        audit.poll(frame);particles.diagnostics->poll(frame);
        const auto& a=audit.latest();const auto& pool=particles.diagnostics->latest();
        if(!a.valid || !pool.valid || a.tick!=pool.tick || a.sequence==lastSample) return;
        lastSample=a.sequence;
        const bool valid=a.active==options.capacity && !a.free && !a.reserved && !a.rejected;
        if(!valid) failure="GPU ownership, numeric state or identity audit failed";
        sawFull=sawFull || pool.active==options.capacity;sawEmpty=sawEmpty || pool.active==0;sawOverflow=sawOverflow || pool.rejected>0;
        if(samples.size()<8192) samples.push_back({{"tick",a.tick},{"elapsed_seconds",elapsed()},{"checked_slots",a.active},
            {"ownership_error",bool(a.free)},{"state_error",bool(a.reserved)},{"identity_error",bool(a.rejected)},
            {"active",pool.active},{"free",pool.free},{"rejected_spawns",pool.rejected}});
        else ++discardedSamples;
    };
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
        spawn.minimum={-6,1,-6};spawn.maximum={6,11,6};
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
        audit.prepare(4,particles.state[particles.current],particles.pool->activeSlots(),particles.pool->freeSlots(),particles.pool->counts(),uint32_t(particles.materials.entries.size()));
        particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),frustum);
        if(elapsed()>=nextSample || phase==0 || phase==1 || phase==129 || phase==130 || phase==131 || phase==215) {
            const bool requested=audit.request(6,ticks);
            const bool poolRequested=particles.diagnostics->request(6,particles.pool->counts(),ticks);
            if(requested!=poolRequested) throw std::runtime_error("Soak snapshot rings lost synchronization");
            if(elapsed()>=nextSample) nextSample=elapsed()+1;
        }
        bgfx::touch(10);particles.render(10,indirect);collect(bgfx::frame());
        if(elapsed()>=nextProgress) {
            std::cout<<"Soak "<<int(elapsed())<<"/"<<options.soakSeconds<<" seconds; tick "<<ticks<<"; audited snapshots "<<samples.size()<<std::endl;
            nextProgress=elapsed()+30;
        }
    }
    while(audit.pending() || particles.diagnostics->pending()) collect(bgfx::frame());
    audit.request(6,ticks);particles.diagnostics->request(6,particles.pool->counts(),ticks);
    while(audit.pending() || particles.diagnostics->pending()) collect(bgfx::frame());
    if(particles.commands.rejected()!=rejectedCommands) failure="Unexpected CPU rejection count";
    if(failure.empty() && (ticks<256 || !sawFull || !sawEmpty || !sawOverflow)) failure="Soak did not complete required full/empty/overflow coverage";
    const auto& finalAudit=audit.latest();const auto& finalPool=particles.diagnostics->latest();
    Json report={{"schema_version",1},{"workload","bounded_256_tick_pool_cycle_v1"},{"seed",42},{"tick_seconds",1.0/60.0},{"audit_every_tick",true},{"passed",failure.empty()},{"failure",failure},{"capacity",options.capacity},
        {"requested_seconds",options.soakSeconds},{"elapsed_seconds",elapsed()},{"ticks",ticks},
        {"slot_checks",ticks*options.capacity},{"completed_cycles",ticks/256},{"spawn_requests",spawnRequests},
        {"intentional_cpu_rejections",rejectedCommands},{"saw_full",sawFull},{"saw_empty",sawEmpty},{"saw_overflow",sawOverflow},
        {"machine",{{"platform",SDL_GetPlatform()},{"logical_cpus",SDL_GetNumLogicalCPUCores()},{"ram_mb",SDL_GetSystemRAM()},{"gpu_vendor_id",caps->vendorId},{"gpu_device_id",caps->deviceId}}},
        {"final_snapshot",{{"tick",finalAudit.tick},{"checked_slots",finalAudit.active},{"ownership_error",bool(finalAudit.free)},{"state_error",bool(finalAudit.reserved)},{"identity_error",bool(finalAudit.rejected)},{"active",finalPool.active},{"free",finalPool.free},{"rejected_spawns",finalPool.rejected}}},
        {"backend",bgfx::getRendererName(caps->rendererType)},{"resolution",{{"width",width},{"height",height}}},
        {"indirect",indirect},{"vsync",false},{"build",build},{"shader_fingerprints",shaderFingerprints},{"materials",materials},{"samples",samples},{"discarded_samples",discardedSamples},
        {"notes",Json::array({"One 1/60-second tick per frame; duration is wall-clock time.","Every tick audits every slot; error flags remain latched across readback intervals.","256-tick cycles cover fill/overflow, duplicate removal, reuse, short lifetimes and impulses.","This is a correctness soak, not a performance benchmark or reference-machine certification."})}};
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
        Graphics graphics(window,!benchmarking && !options.soakSeconds);
        const auto* caps=bgfx::getCaps();
        const uint64_t required=BGFX_CAPS_COMPUTE;
        std::cout << "Backend: " << bgfx::getRendererName(caps->rendererType) << '\n';
        if((caps->supported&required)!=required || caps->limits.maxInstanceData < 2 || caps->limits.maxComputeBindings < 6) throw std::runtime_error("Required compute/instancing unavailable; cannot run GPU simulation");
        std::cout << "Compute and instancing supported; indirect draws "
            << ((caps->supported&BGFX_CAPS_DRAW_INDIRECT)?"available":"unavailable") << '\n';
        std::cout << "Submission: compact visible instances; workgroup: 128\n";
        const char* basePath=SDL_GetBasePath();
        if(!basePath) throw std::runtime_error("SDL cannot resolve application assets");
        const std::filesystem::path assets=std::filesystem::path(basePath)/"assets";
        pw::runGpuLayoutTest(assets);
        pw::runGpuPoolTest(assets);
        pw::runGpuVisibilityTest(assets);
        pw::runGpuAuditTest(assets);
        if(gpuTestOnly) return 0;
        Particles particles(assets,options.capacity);
        if(options.soakSeconds) return runSoak(window,particles,assets,options);
        if(benchmarking) return runBenchmark(window,particles,assets,options);
        GuiContext gui(window.get());
        pw::ImGuiRenderer guiRenderer(assets);

        pw::FixedStep clock;
        bool running=true,paused=false,cull=true;
        bool indirect=allowIndirect && particles.visibility->supportsIndirect();
        std::cout<<"Drawing: "<<(indirect?"GPU indirect":"masked fixed-capacity")<<'\n';
        uint64_t ticks=0; uint32_t frames=0, completedFrame=0; double lag=0;
        int width=0,height=0;
        float yaw=0.65f, distance=24, elevation=12, impulseStrength=6, spawnLifetime=10;
        int spawnCount=1024;
        auto previous=std::chrono::steady_clock::now();
        while(running) {
            particles.diagnostics->poll(completedFrame);
            particles.visibleDiagnostics->poll(completedFrame);
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
            ImGui::SetNextWindowSize(ImVec2(460,680),ImGuiCond_FirstUseEver);
            ImGui::Begin("Particle World - foundation");
            ImGui::Text("Backend: %s",bgfx::getRendererName(caps->rendererType));
            ImGui::Text("Capacity: %u | Seed: 42",particles.particleCount);
            const auto& poolStats=particles.diagnostics->latest();
            if(poolStats.valid) {
                ImGui::Text("Active: %u | Free: %u",poolStats.active,poolStats.free);
                ImGui::Text("Rejected spawns: %u",poolStats.rejected);
                ImGui::TextDisabled("GPU snapshot at tick %llu",static_cast<unsigned long long>(poolStats.tick));
            } else ImGui::TextDisabled("Waiting for GPU pool snapshot...");
            const auto& visibleStats=particles.visibleDiagnostics->latest();
            if(visibleStats.valid) ImGui::Text("Visible (delayed): %u",visibleStats.active);
            else ImGui::TextDisabled("Waiting for visibility snapshot...");
            ImGui::Text("Tick: %llu",static_cast<unsigned long long>(ticks));
            ImGui::Checkbox("Paused",&paused); ImGui::SameLine();
            bool step=ImGui::Button("Single tick"); if(step) paused=true;
            if(ImGui::Button("Reset scene")) {particles.reset();clock.reset();ticks=0;lag=0;}
            ImGui::SameLine();
            if(ImGui::Button("Empty scene")) {particles.reset(0);clock.reset();ticks=0;lag=0;}
            const char* sceneLabel=particles.sceneMaterial<0?"Mixed solids":particles.materials.entries[particles.sceneMaterial].name.c_str();
            if(ImGui::BeginCombo("Scene material",sceneLabel)) {
                if(ImGui::Selectable("Mixed solids",particles.sceneMaterial<0)) {particles.sceneMaterial=-1;particles.reset();clock.reset();ticks=0;}
                for(std::uint32_t id=0;id<particles.materials.entries.size();++id) {
                    const auto& material=particles.materials.entries[id];
                    if(material.gpu.identity[1]!=0) continue;
                    if(ImGui::Selectable(material.name.c_str(),particles.sceneMaterial==int(id))) {particles.sceneMaterial=int(id);particles.reset();clock.reset();ticks=0;}
                }
                ImGui::EndCombo();
            }
            if(ImGui::Button("Reload materials")) particles.reloadPending=true;
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Reloads %s",particles.materialPath.string().c_str());
            if(particles.reloadPending) ImGui::TextWrapped("Reload queued for next tick (step when paused).");
            if(!particles.materialMessage.empty()) ImGui::TextWrapped("%s",particles.materialMessage.c_str());
            ImGui::Text("Profiles: %u",unsigned(particles.materials.entries.size()));
            ImGui::SliderFloat("Lift speed (m/s)",&impulseStrength,0,20);
            if(ImGui::Button("Lift center particles")) particles.queueBrush(pw::CommandType::ApplyImpulse,impulseStrength);
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Adds upward velocity in the center 8 x 12 x 8 m box. Mixed scenes target Stone.");
            if(ImGui::Button("Remove center particles")) particles.queueBrush(pw::CommandType::RemoveBrush,0);
            ImGui::SliderInt("Spawn count",&spawnCount,1,4096);
            ImGui::SliderFloat("Lifetime (s)",&spawnLifetime,0,30);
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Zero keeps particles alive until removed; positive values expire after that many simulation seconds.");
            if(ImGui::Button("Spawn particles")) particles.queueBrush(pw::CommandType::SpawnBrush,0,uint32_t(spawnCount),spawnLifetime);
            ImGui::Text("Queued: %u | Rejected commands: %llu",unsigned(particles.commands.pending()),static_cast<unsigned long long>(particles.commands.rejected()));
            if(!particles.commandMessage.empty()) ImGui::TextWrapped("%s",particles.commandMessage.c_str());
            ImGui::Checkbox("Frustum culling",&cull);
            if(particles.visibility->supportsIndirect()) {ImGui::SameLine();ImGui::Checkbox("Indirect draw",&indirect);}
            ImGui::SliderFloat("Orbit",&yaw,-3.14f,3.14f);
            ImGui::SliderFloat("Distance",&distance,8,45);
            ImGui::SliderFloat("Elevation",&elevation,2,25);
            const auto* stats=bgfx::getStats();
            if(stats->cpuTimerFreq) ImGui::Text("CPU frame: %.2f ms",1000.0*double(stats->cpuTimeFrame)/double(stats->cpuTimerFreq));
            if(stats->gpuTimerFreq) ImGui::Text("GPU frame: %.2f ms",1000.0*double(stats->gpuTimeEnd-stats->gpuTimeBegin)/double(stats->gpuTimerFreq));
            ImGui::Text("Dropped wall time: %.3f s",lag);
            ImGui::TextWrapped("GPU pool: spawn, remove, lifetime and box contacts. Pair collisions are pending.");
            ImGui::End();
            auto batch=clock.advance(elapsed,paused,step); lag+=batch.droppedSeconds;
            for(unsigned i=0;i<batch.count;++i) {particles.tick(uint16_t(i+1));++ticks;}
            if(frames%8==0 || step || !particles.diagnostics->latest().valid)
                particles.diagnostics->request(6,particles.pool->counts(),ticks);
            if(frames%8==0 || step || !particles.visibleDiagnostics->latest().valid)
                particles.visibleDiagnostics->request(6,particles.visibility->counts(),ticks);
            float view[16],projection[16];
            bx::mtxLookAt(view,{std::sin(yaw)*distance,elevation,std::cos(yaw)*distance},{0,3,0},{0,1,0},bx::Handedness::Right);
            bx::mtxProj(projection,55,float(width)/float(height),0.1f,150,caps->homogeneousDepth,bx::Handedness::Right);
            float viewProjection[16];bx::mtxMul(viewProjection,view,projection);
            particles.visibility->prepare(5,particles.instances,particles.pool->activeSlots(),particles.pool->counts(),pw::extractFrustum(viewProjection,caps->homogeneousDepth),cull);
            bgfx::setViewRect(10,0,0,uint16_t(width),uint16_t(height));
            bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x101925ff,1.0f,0);
            bgfx::setViewTransform(10,view,projection); bgfx::touch(10);
            particles.render(10,indirect);
            ImGui::Render(); guiRenderer.render(ImGui::GetDrawData(),20);
            completedFrame=bgfx::frame();
            ++frames;
            if(frameLimit && frames>=frameLimit) running=false;
        }
        std::cout << "Clean shutdown after " << ticks << " ticks\n";
        return 0;
    } catch(const std::exception& error) {std::cerr << "Particle World: " << error.what() << '\n';return 1;}
}
