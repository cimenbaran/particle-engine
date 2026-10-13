#include "GpuSettling.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "assets/Materials.hpp"
#include "../../tests/reference/SettlingSupport.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct Handle {
    T value;explicit Handle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Settling test resource unavailable");}
    ~Handle(){bgfx::destroy(value);}Handle(const Handle&)=delete;
};
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
}
void runGpuSettlingTest(const std::filesystem::path& assets) {
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,3> materials{};for(unsigned i=0;i<3;++i){materials[i].identity[0]=i;materials[i].identity[1]=i;}
    Handle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    Handle exporter(loadCompute(assets,"cs_export")),statExporter(loadCompute(assets,"cs_pool_stats"));
    Handle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned cases=0;
    auto run=[&](std::vector<ParticleGpu> particles,unsigned ticks=1,GpuSettling* reused=nullptr,
                 std::vector<SettlingState>* reusedHistory=nullptr,unsigned failure=0,unsigned upstreamError=0,SettlingProfile profile=SettlingProfile{},AnalyticSphere obstacle={}) {
        const auto count=std::uint32_t(particles.size()),storage=std::max(1u,count);
        auto owned=reused?nullptr:std::make_unique<GpuSettling>(assets,count,profile,false,obstacle);auto& tracker=reused?*reused:*owned;
        std::vector<SettlingState> local(count);auto& expected=reusedHistory?*reusedHistory:local;
        auto uploaded=particles;if(!count)uploaded.resize(1);
        Handle state(bgfx::createDynamicVertexBuffer(bgfx::copy(uploaded.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        const std::array<std::uint32_t,4> upstream{upstreamError,0,0,0};
        Handle sourceStatus(bgfx::createDynamicIndexBuffer(bgfx::copy(upstream.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        std::vector<reference::SettlingBody> bodies(count);
        for(unsigned i=0;i<count;++i){const auto& p=particles[i];auto& b=bodies[i];for(unsigned a=0;a<3;++a)b.position[a]=p.positionRadius[a];b.alive=(p.identity[1]&1)!=0;b.solid=p.identity[0]==0;b.dynamic=p.velocityInvMass[3]>0;}
        for(unsigned tick=0;tick<ticks;++tick){
            tracker.submit(0,state.value,materialBuffer.value,3,sourceStatus.value);
            for(unsigned i=0;i<count;++i){
                const auto& p=particles[i];auto support=failure?reference::SettlingSupport{}:reference::settlingSupport(bodies,i,{obstacle.centre[0],obstacle.centre[1],obstacle.centre[2]},obstacle.radius);
                // One-axis threshold fixtures have exact speed; general geometry
                // fixtures stay away from numerical classification boundaries.
                float speed=std::hypot(p.velocityInvMass[0],p.velocityInvMass[1],p.velocityInvMass[2]);
                expected[i]=advanceSettling(expected[i],{p.identity[3],speed,bodies[i].alive&&bodies[i].solid&&bodies[i].dynamic,support.enter,support.hold,failure==0},profile);
            }
            bgfx::frame();SDL_PumpEvents();
        }
        Handle image(bgfx::createTexture2D(1,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle staging(bgfx::createTexture2D(1,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        Handle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        Handle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count){
            const float h[]{0,float(count),0,1},s[]{0,float(count),0,4};
            bgfx::setUniform(uniform.value,h);bgfx::setBuffer(0,tracker.history(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            bgfx::setUniform(uniform.value,s);bgfx::setBuffer(0,state.value,bgfx::Access::Read);bgfx::setImage(1,oldImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,tracker.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statExporter.value,1);
        std::vector<SettlingState> actual(storage);std::vector<ParticleGpu> original(storage);std::array<std::uint32_t,4> status{};
        bgfx::TextureRegion src,dst;src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,status.data());
        if(count){src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if(status[0]!=failure)throw std::runtime_error("Settling fixture "+std::to_string(cases)+" error mask "+std::to_string(status[0])+" expected "+std::to_string(failure));
        unsigned candidates=0,acquiring=0,unsupported=0;
        for(unsigned i=0;i<count;++i){
            if(actual[i]!=expected[i])throw std::runtime_error("Settling fixture "+std::to_string(cases)+" history mismatch slot "+std::to_string(i)+" got "+std::to_string(actual[i].quietTicks)+","+std::to_string(actual[i].unsupportedTicks)+","+std::to_string(actual[i].candidate)+" expected "+std::to_string(expected[i].quietTicks)+","+std::to_string(expected[i].unsupportedTicks)+","+std::to_string(expected[i].candidate));
            require(std::memcmp(&original[i],&particles[i],sizeof(ParticleGpu))==0,"Settling modified particle state/active flags");
            candidates+=expected[i].candidate;acquiring+=!expected[i].candidate && expected[i].quietTicks>0;unsupported+=expected[i].candidate && expected[i].unsupportedTicks>0;
        }
        require(status[1]==candidates && status[2]==acquiring && status[3]==unsupported,"Settling count mismatch");++cases;
    };
    auto fixture=[](unsigned count){std::vector<ParticleGpu> p(count);for(unsigned i=0;i<count;++i)p[i]={{-4+float(i%32)*0.25f,0.0625f,-4+float(i/32)*0.25f,0.0625f},{0,0,0,1},{300,0,0,0},{0,1,i,1}};return p;};
    for(unsigned n:{0u,1u,127u,128u,129u,255u,256u,257u,1024u})run(fixture(n));
    GpuSettling tracker(assets,129);std::vector<SettlingState> history(129);auto p=fixture(129);
    run(p,29,&tracker,&history);require(!history[0].candidate,"Settling dwell acquired early");
    run(p,1,&tracker,&history);require(history[0].candidate,"Settling dwell failed");
    for(auto& body:p){body.positionRadius[1]+=0.001f;body.velocityInvMass[0]=0.075f;}
    run(p,8,&tracker,&history);require(history[0].candidate,"Settling dead band lost history");
    for(auto& body:p)body.positionRadius[1]=1;
    run(p,2,&tracker,&history);require(history[0].candidate && history[0].unsupportedTicks==2,"Settling grace failed");
    run(p,1,&tracker,&history);require(!history[0].candidate,"Settling missing support did not release");
    p=fixture(129);run(p,30,&tracker,&history);
    p[0].velocityInvMass[0]=0.11f;p[1].identity[3]=2;p[2].identity[1]=0;p[3].identity[0]=1;p[4].velocityInvMass[3]=0;
    run(p,1,&tracker,&history);require(!history[0].candidate && history[1].quietTicks==1 && !history[2].candidate,"Settling impulse/recycling/exclusion failed");
    tracker.reset(0);history.assign(129,{});p=fixture(129);run(p,1,&tracker,&history);require(history[0].quietTicks==1,"Settling reset retained history");
    run(p,30,&tracker,&history);run(p,1,&tracker,&history,1,1);run(p,1,&tracker,&history);require(history[0].quietTicks==1,"Settling upstream fault retained stale history");
    p[0].positionRadius[3]=0.05f;run(p,1,&tracker,&history,4);p=fixture(129);run(p,1,&tracker,&history);
    p[128].positionRadius[0]=9;run(p,1,&tracker,&history,2);
    p=fixture(129);p[7].velocityInvMass[0]=std::numeric_limits<float>::quiet_NaN();run(p,1,&tracker,&history,4);
    p=fixture(129);p[7].velocityInvMass[3]=-1;run(p,1,&tracker,&history,4);
    p=fixture(129);p[7].identity[3]=0;run(p,1,&tracker,&history,2);
    p=fixture(129);for(auto& body:p)body.positionRadius[1]=0.05f;run(p,30);
    // Stack, static support, side-only contact, coincident and deeply overlapping
    // pairs, and excluded phases; all support compared to the double oracle.
    p=fixture(16);for(unsigned i=0;i<p.size();++i){p[i].positionRadius[0]=0;p[i].positionRadius[1]=0.0625f+i*0.125f;p[i].positionRadius[2]=0;}p[0].velocityInvMass[3]=0;run(p,30);
    p=fixture(4);for(auto& body:p){body.positionRadius[0]=0;body.positionRadius[1]=8;}run(p,30);
    p=fixture(2);p[0].positionRadius[1]=8;p[1].positionRadius[1]=8.1f;p[1].positionRadius[0]=p[0].positionRadius[0];run(p,30);
    p=fixture(3);for(auto& body:p)body.positionRadius[1]=8;p[1].positionRadius[0]=p[0].positionRadius[0];p[1].positionRadius[1]+=0.125f;p[0].identity[0]=2;run(p,30);
    p=fixture(129);for(unsigned i=0;i<p.size();++i){p[i].velocityInvMass[0]=i%2?0.05f:0.1f;}run(p,30);
    p=fixture(1);p[0].velocityInvMass[0]=std::numeric_limits<float>::max();run(p);
    SettlingProfile immediate;immediate.acquireTicks=1;immediate.releaseTicks=1;run(fixture(129),1,nullptr,nullptr,0,0,immediate);
    p=fixture(1024);for(unsigned i=0;i<p.size();++i){p[i].positionRadius[0]=0;p[i].positionRadius[1]=i<512?1.0f:1.125f;p[i].positionRadius[2]=0;}
    run(p,1,nullptr,nullptr,0,0,immediate); // 1023 accepted neighbors, no cap.
    run({},1,nullptr,nullptr,1,1);
    SettlingProfile maximumProfile;maximumProfile.acquireTicks=std::numeric_limits<std::uint32_t>::max();maximumProfile.releaseTicks=maximumProfile.acquireTicks;
    GpuSettling maximum(assets,1,maximumProfile);maximum.reset(0);bgfx::frame();
    std::vector<SettlingState> maximumHistory{{1,maximumProfile.acquireTicks-1,0,0}};
    bgfx::update(maximum.history(),0,bgfx::copy(maximumHistory.data(),sizeof(SettlingState)));
    run(fixture(1),1,&maximum,&maximumHistory,0,0,maximumProfile);require(maximumHistory[0].candidate,"Settling maximum dwell wrapped");
    maximumHistory[0].unsupportedTicks=maximumProfile.releaseTicks-1;
    bgfx::update(maximum.history(),0,bgfx::copy(maximumHistory.data(),sizeof(SettlingState)));
    p=fixture(1);p[0].positionRadius[1]=1;run(p,1,&maximum,&maximumHistory,0,0,maximumProfile);
    require(!maximumHistory[0].candidate,"Settling maximum grace wrapped");
    GpuSettling latched(assets,1,{},true);std::vector<SettlingState> latchedHistory(1);
    run(fixture(1),30,&latched,&latchedHistory);
    run(fixture(1),1,&latched,&latchedHistory,1,1);
    run(fixture(1),30,&latched,&latchedHistory,1);
    latched.reset(0);latchedHistory.assign(1,{});run(fixture(1),1,&latched,&latchedHistory);
    require(latchedHistory[0].quietTicks==1,"Settling latch reset failed");
    GpuSettling emptyLatched(assets,0,{},true);std::vector<SettlingState> noHistory;
    run({},1,&emptyLatched,&noHistory,1,1);run({},1,&emptyLatched,&noHistory,1);
    emptyLatched.reset(0);run({},1,&emptyLatched,&noHistory);
    const AnalyticSphere sphere{{0,4,0},0.9375f};
    // Surface normals above/below the support cone, entry/hold gaps, and deep overlap.
    for(float normalY:{1.0f,0.8f,0.501f,0.499f,0.0f,-1.0f})for(float gap:{0.0f,0.00005f,0.001f,0.003f,-0.01f}) {
        p=fixture(1);p[0].positionRadius[0]=std::sqrt(1-normalY*normalY)*(1+gap);
        p[0].positionRadius[1]=4+normalY*(1+gap);p[0].positionRadius[2]=0;
        run(p,30,nullptr,nullptr,0,0,{},sphere);
    }
    GpuSettling curved(assets,1,{},false,sphere);std::vector<SettlingState> curvedHistory(1);
    p=fixture(1);p[0].positionRadius[0]=0;p[0].positionRadius[1]=5;p[0].positionRadius[2]=0;
    run(p,30,&curved,&curvedHistory,0,0,{},sphere);require(curvedHistory[0].candidate,"Sphere support failed to acquire");
    p[0].positionRadius[1]=5.001f;run(p,8,&curved,&curvedHistory,0,0,{},sphere);require(curvedHistory[0].candidate,"Sphere hold gap lost candidate");
    p[0].positionRadius[1]=5.003f;run(p,2,&curved,&curvedHistory,0,0,{},sphere);require(curvedHistory[0].candidate,"Sphere grace released early");
    run(p,1,&curved,&curvedHistory,0,0,{},sphere);require(!curvedHistory[0].candidate,"Sphere unsupported candidate retained");
    p[0].positionRadius[1]=5;run(p,30,&curved,&curvedHistory,0,0,{},sphere);
    run(p,1,&curved,&curvedHistory,1,1,{},sphere);require(!curvedHistory[0].candidate,"Sphere upstream failure retained candidate");
    curved.reset(0);curvedHistory.assign(1,{});run(p,1,&curved,&curvedHistory,0,0,{},sphere);
    for(unsigned n:{127u,128u,129u,255u,256u,257u}) {
        p=fixture(n);for(auto& body:p){body.positionRadius[0]=0;body.positionRadius[1]=5;body.positionRadius[2]=0;}
        run(p,1,nullptr,nullptr,0,0,immediate,sphere);
    }
    bool rejected=false;try{SettlingProfile invalid;invalid.enterSpeed=0;GpuSettling bad(assets,1,invalid);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"GPU settling accepted unsupported speed profile");
    std::cout<<"GPU settling passed: "<<cases<<" fixtures, exact per-slot CPU histories, support geometry, reset/recycling/faults and unchanged particle bytes\n";
}
}
