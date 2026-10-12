#include "ParticleAudit.hpp"
#include "core/ParticleTypes.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <limits>
#include <iostream>
#include <stdexcept>
namespace pw {
namespace {
template<class T> struct Owned {T value;explicit Owned(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Audit fixture allocation failed");}~Owned(){bgfx::destroy(value);}Owned(const Owned&)=delete;};
}
void runGpuAuditTest(const std::filesystem::path& assets) {
    ParticleAudit audit(assets,4);
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    // Clean, duplicate, bad index, NaN, infinity, stale ID, zero generation, bad material, and empty.
    for(unsigned fault=0;fault<10;++fault) {
        std::array<ParticleGpu,4> particles{};
        for(unsigned i=0;i<4;++i) particles[i]={{0,1,0,.045f},{0,0,0,1},{293.15f,0,0,0},{0,i<2?1u:0u,i,1}};
        std::array<uint32_t,4> active{0,1,0,0},free{2,3,0,0},counts{2,2,0,0};
        if(fault==1) active[1]=0;
        if(fault==2) free[0]=0xffffffffu;
        if(fault==3) particles[0].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();
        if(fault==4) particles[0].velocityInvMass[1]=std::numeric_limits<float>::infinity();
        if(fault==5) particles[0].identity[2]=3;
        if(fault==6) particles[0].identity[3]=0;
        if(fault==7) particles[0].identity[0]=7;
        if(fault==8) {counts={0,4,0,0};free={0,1,2,3};for(auto& p:particles) p.identity[1]=0;}
        if(fault==9) particles[3].velocityInvMass[0]=std::numeric_limits<float>::max();
        Owned state(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),sizeof(particles)),lanes,BGFX_BUFFER_COMPUTE_READ));
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ;
        Owned activeBuffer(bgfx::createDynamicIndexBuffer(bgfx::copy(active.data(),sizeof(active)),flags));
        Owned freeBuffer(bgfx::createDynamicIndexBuffer(bgfx::copy(free.data(),sizeof(free)),flags));
        Owned pool(bgfx::createDynamicIndexBuffer(bgfx::copy(counts.data(),sizeof(counts)),flags));
        auto check=[&](bool expectedOwnership,bool expectedState,bool expectedId) {
            audit.prepare(0,state.value,activeBuffer.value,freeBuffer.value,pool.value,7);
            if(!audit.request(1,fault)) throw std::runtime_error("Audit ring full");
            while(audit.pending()) {audit.poll(bgfx::frame());SDL_PumpEvents();}
            const auto& result=audit.latest();
            if(!result.valid || result.active!=4 || bool(result.free)!=expectedOwnership || bool(result.reserved)!=expectedState || bool(result.rejected)!=expectedId)
                throw std::runtime_error("GPU audit did not detect expected fault "+std::to_string(fault));
        };
        audit.reset();check(fault==1 || fault==2,fault==3 || fault==4,fault>=5 && fault<=7);
        // The next clean epoch clears flags; the same epoch must retain an observed fault.
        if(fault==1) {
            active={0,1,0,0};bgfx::update(activeBuffer.value,0,bgfx::copy(active.data(),sizeof(active)));
            check(true,false,false);
            audit.reset();check(false,false,false);
        }
    }
    std::cout<<"GPU audit clean/empty, injected ownership/numeric/identity faults and error latching passed\n";
}
}
