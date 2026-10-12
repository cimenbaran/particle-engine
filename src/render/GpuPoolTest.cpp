#include "GpuPoolTest.hpp"
#include "ParticlePool.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct Owned {
    T value;
    explicit Owned(T h):value(h) {if(!bgfx::isValid(h)) throw std::runtime_error("Pool test resource unavailable");}
    ~Owned(){bgfx::destroy(value);}
    Owned(const Owned&)=delete;
};
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
}
void runGpuPoolTest(const std::filesystem::path& assets) {
    Owned reset(loadCompute(assets,"cs_reset"));
    Owned integrate(loadCompute(assets,"cs_particles"));
    Owned exportState(loadCompute(assets,"cs_export"));
    Owned exportPool(loadCompute(assets,"cs_pool_export"));
    Owned sim(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    Owned poolUniform(bgfx::createUniform("u_pool",bgfx::UniformType::Vec4));
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    const auto library=loadMaterials(assets/"materials"/"default.json");const auto packed=library.packed();
    Owned materials(bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),uint32_t(packed.size()*sizeof(MaterialGpu))),lanes,BGFX_BUFFER_COMPUTE_READ));
    for(uint32_t capacity:{1u,127u,128u,129u,16384u}) {
        ParticlePool pool(assets,capacity);
        Owned state(bgfx::createDynamicVertexBuffer(capacity*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Owned next(bgfx::createDynamicVertexBuffer(capacity*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Owned instances(bgfx::createDynamicVertexBuffer(capacity*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        std::vector<ParticleGpu> initial(capacity),particles(capacity);
        for(uint32_t i=0;i<capacity;++i) initial[i].identity[2]=i;
        Owned upload(bgfx::createDynamicVertexBuffer(bgfx::copy(initial.data(),capacity*sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
        const float simParams[]{0,float(capacity),float(packed.size()),0};
        bgfx::setViewMode(0,bgfx::ViewMode::Sequential);
        bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,upload.value,bgfx::Access::Read);
        bgfx::setBuffer(1,state.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,reset.value,(capacity+127)/128);
        pool.rebuild(0,state.value,true);
        const uint16_t rows=uint16_t((capacity+1+255)/256);
        Owned out(bgfx::createTexture2D(256,rows,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned staging(bgfx::createTexture2D(256,rows,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        Owned particleOut(bgfx::createTexture2D(4,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned particleStaging(bgfx::createTexture2D(4,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<std::array<uint32_t,4>> membership(uint32_t(rows)*256);
        auto check=[&](uint32_t alive,uint32_t rejected) {
            const float params[]{float(capacity),0,0,0};bgfx::setUniform(poolUniform.value,params);
            bgfx::setBuffer(0,pool.counts(),bgfx::Access::Read);bgfx::setBuffer(1,pool.freeSlots(),bgfx::Access::Read);
            bgfx::setBuffer(2,pool.activeSlots(),bgfx::Access::Read);
            bgfx::setImage(3,out.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exportPool.value,(capacity+1+127)/128);
            bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
            bgfx::setImage(1,particleOut.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exportState.value,(capacity+127)/128);
            bgfx::TextureRegion dst;dst.handle=staging.value;bgfx::TextureRegion src;src.handle=out.value;
            bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,membership.data());
            dst.handle=particleStaging.value;src.handle=particleOut.value;
            bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,particles.data()));
            while(bgfx::frame()<ready) SDL_PumpEvents();
            const auto counts=membership[capacity];
            require(counts[0]==alive && counts[1]==capacity-alive && counts[3]==rejected,"Pool count/overflow mismatch");
            std::vector<unsigned> owners(capacity);
            for(unsigned kind=0;kind<2;++kind) {
                const auto count=kind==0?counts[1]:counts[0];
                for(uint32_t i=0;i<count;++i) {
                    const auto slot=membership[i][kind];
                    require(slot<capacity,"Pool slot out of capacity");
                    require(++owners[slot]==1,"Duplicate pool owner");
                    require(bool(particles[slot].identity[1]&1u)==bool(kind),"Pool owner/alive disagreement");
                    require(particles[slot].identity[2]==slot,"Stable slot ID changed");
                }
            }
            for(auto ownersForSlot:owners) require(ownersForSlot==1,"Orphan pool slot");
        };
        check(0,0);
        Command spawn;spawn.type=CommandType::SpawnBrush;spawn.minimum={-1,5,-1};spawn.maximum={1,6,1};
        spawn.parameters={0,0,0,0};spawn.count=1;
        uint64_t sequence=0;
        auto apply=[&](Command command) {command.sequence=sequence++;pool.apply(0,state.value,materials.value,uint32_t(packed.size()),command);};
        // Zero requests are a no-op; then fill exactly to capacity across bounded batches.
        spawn.count=0;apply(spawn);pool.rebuild(0,state.value);check(0,0);
        for(uint32_t remaining=capacity;remaining;) {spawn.count=std::min(remaining,4096u);apply(spawn);remaining-=spawn.count;}
        pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) {
            require(p.identity[3]==1,"Initial generation mismatch");
            require(p.positionRadius[0]>=-1 && p.positionRadius[0]<=1 && p.positionRadius[1]>=5 && p.positionRadius[1]<=6,"Spawn escaped brush");
            require(std::isfinite(p.velocityInvMass[3]) && p.velocityInvMass[3]>0,"Spawn mass invalid");
        }
        Command impulse;impulse.type=CommandType::ApplyImpulse;impulse.minimum={-1,5,-1};impulse.maximum={1,6,1};impulse.parameters={0.25f,1,-0.5f,0};
        apply(impulse);pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) require(p.velocityInvMass[0]==0.25f && p.velocityInvMass[1]==1 && p.velocityInvMass[2]==-0.5f,"Pool impulse mismatch");
        for(unsigned i=1;i<32;++i) apply(impulse);
        // Spatial miss and material miss must preserve the velocity exactly.
        impulse.minimum={2,5,2};impulse.maximum={3,6,3};apply(impulse);
        impulse.minimum={-1,5,-1};impulse.maximum={1,6,1};impulse.material=1;apply(impulse);
        pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) require(p.velocityInvMass[0]==8 && p.velocityInvMass[1]==32 && p.velocityInvMass[2]==-16,"Ordered pool impulse/filter mismatch");
        spawn.count=127;apply(spawn);pool.rebuild(0,state.value);check(capacity,127);
        Command remove;remove.type=CommandType::RemoveBrush;remove.minimum={-8,0,-8};remove.maximum={8,12,8};
        apply(remove);apply(remove);pool.rebuild(0,state.value);check(0,127);
        spawn.count=std::min(capacity,128u);spawn.parameters[3]=1.0f/60.0f;apply(spawn);
        pool.rebuild(0,state.value);check(spawn.count,127);
        for(const auto& p:particles) if(p.identity[1]&1u) require(p.identity[3]==2,"Reused slot generation did not advance");
        // Expiry uses the regular integration path and masks the fixed-capacity render record.
        const float step[]{1.0f/60.0f,float(capacity),float(packed.size()),0};
        bgfx::setUniform(sim.value,step);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setBuffer(1,next.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,integrate.value,(capacity+127)/128);
        std::swap(state.value,next.value);pool.rebuild(0,state.value);check(0,127);
        // Export the first two lanes of each texture row to verify dead render records.
        const float renderParams[]{0,float(capacity),0,2};
        bgfx::setUniform(sim.value,renderParams);bgfx::setBuffer(0,instances.value,bgfx::Access::Read);
        bgfx::setImage(1,particleOut.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exportState.value,(capacity+127)/128);
        bgfx::TextureRegion renderDst;renderDst.handle=particleStaging.value;
        bgfx::TextureRegion renderSrc;renderSrc.handle=particleOut.value;
        bgfx::blit(2,renderDst,renderSrc);
        std::vector<ParticleGpu> rendered(capacity);
        const auto renderReady=bgfx::read(renderDst,rendered.data());
        while(bgfx::frame()<renderReady) SDL_PumpEvents();
        for(const auto& record:rendered) for(unsigned field=0;field<4;++field)
            require(record.positionRadius[field]==0 && record.velocityInvMass[field]==0,"Dead instance was not masked");
        if(capacity==1) {
            // Exhausted generations must never wrap to a previously valid handle.
            particles[0].identity[3]=0xffffffffu;
            Owned exhausted(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
            bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,exhausted.value,bgfx::Access::Read);
            bgfx::setBuffer(1,state.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
            bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,reset.value,1);
            pool.rebuild(0,state.value,true);spawn.count=1;apply(spawn);pool.rebuild(0,state.value);check(0,1);
            require(particles[0].identity[3]==0xffffffffu,"Slot generation wrapped");
        }
        std::cout<<"GPU pool ownership/full/duplicate-remove/reuse/lifetime passed: "<<capacity<<" slots\n";
    }
}
}
