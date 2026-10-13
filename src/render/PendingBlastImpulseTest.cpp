#include "PendingBlastImpulse.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/TerrainMass.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
namespace pw {
namespace {
template<class T>struct ImpulseHandle {T value;explicit ImpulseHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Impulse test allocation failed");}~ImpulseHandle(){bgfx::destroy(value);}ImpulseHandle(const ImpulseHandle&)=delete;};
}
void runPendingBlastImpulseTest(const std::filesystem::path& assets) {
    using State=PendingBlastImpulse::State;using Store=terrain::SparseVoxelStore;
    const auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    const auto rejects=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad,"Invalid pending impulse accepted");};
    const auto materials=loadMaterials(assets/"materials/default.json");
    const auto world=[&]{auto store=std::make_unique<Store>(terrain::ResidentChunkLayout({0,0,0},{1,1,1}),unsigned(materials.entries.size()));store->setMaterial(*terrain::addressOf({8,8,8}),0,0);return VoxelPublicationPtr(std::make_shared<VoxelPublication>(assets,std::move(store),terrain::Chunk{},materials));};
    ImpulseHandle exporter(loadCompute(assets,"cs_export"));ImpulseHandle params(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    ImpulseHandle image(bgfx::createTexture2D(4,VoxelProbes::poolCapacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
    ImpulseHandle staging(bgfx::createTexture2D(4,VoxelProbes::poolCapacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
    const auto read=[&](VoxelProbes& particles,bool history=false) {
        const float p[]{0,float(particles.capacity()),0,history?1.f:4.f};bgfx::setUniform(params.value,p);bgfx::setBuffer(0,history?particles.settlingHistory():particles.state(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,exporter.value,dispatchGroups(particles.capacity()));
        bgfx::TextureRegion a,b;a.handle=image.value;b.handle=staging.value;bgfx::blit(7,b,a);std::vector<ParticleGpu> result(particles.capacity());const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
    };
    const auto populate=[&](VoxelProbes& particles) {auto input=read(particles);const auto inv=terrain::conversionMassProfile(materials.entries[0].gpu.mechanical[0]).inverseMass;for(unsigned i=0;i<2;++i)input[i]={{1.0625f+float(i)*.375f,2.0625f,1.0625f,.0625f},{0,0,0,inv},{293.15f,0,0,0},{0,1,i,1}};bgfx::update(particles.state(),0,bgfx::copy(input.data(),unsigned(input.size()*64)));return input;};
    const auto wait=[&](PendingBlastImpulse& p){for(unsigned i=0;i<128 && p.state()==State::Waiting;++i){p.poll(bgfx::frame());SDL_PumpEvents();}require(p.state()!=State::Waiting,"Impulse acknowledgement timed out");};
    BlastImpulseRequest request{{1.0625f,2.0625f,1.0625f},1,2,100};unsigned cases=0;
    {
        auto live=world(),original=live;VoxelProbes particles(assets,materials,true,true);const auto before=populate(particles);
        const std::array<std::uint32_t,4> settled{1,30,0,1};std::vector<std::array<std::uint32_t,4>> history(particles.capacity(),settled);
        bgfx::update(particles.settlingHistory(),0,bgfx::copy(history.data(),unsigned(history.size()*16)));const auto oldHistory=read(particles,true);
        require(std::memcmp(oldHistory[0].positionRadius,settled.data(),16)==0,"Settling history fixture was not uploaded");
        auto pending=std::make_unique<PendingBlastImpulse>(assets,particles,live,materials,request);
        require(particles.transactionPending() && !pending->publish(0,particles,live) && pending->energyJoules()==0,"Unacknowledged impulse adopted or charged");
        rejects([&]{PendingBlastImpulse duplicate(assets,particles,live,materials,request);});rejects([&]{pending->submit(UINT16_MAX,particles);});
        Command clear;clear.type=CommandType::RemoveBrush;clear.material=0;clear.minimum={-16,0,-16};clear.maximum={16,16,16};require(particles.enqueue(clear).empty(),"Frozen impulse rejected queued command");particles.tick(1,live->occupancy(),live->source());require(particles.ticks()==0 && particles.pendingCommands()==1,"Impulse failed to freeze commands/tick");
        pending->submit(8,particles);particles.prepare(5);particles.request(6);wait(*pending);
        require(pending->state()==State::Ready && pending->accepted()==2 && pending->energyJoules()>0 && pending->energyJoules()<=request.joules,"Valid impulse failed acknowledgement");
        const auto prepared=read(particles);require(std::memcmp(before.data(),prepared.data(),before.size()*64)==0,"Private impulse changed live state");
        rejects([&]{pending->publish(UINT16_MAX,particles,live);});const double charge=pending->energyJoules();
        require(pending->publish(0,particles,live) && !pending->publish(0,particles,live) && !particles.transactionPending() && live==original,"Impulse adoption repeated, failed, or changed terrain");pending.reset();
        particles.request(6);const auto after=read(particles);particles.poll(bgfx::frame());double kinetic=0;
        for(unsigned i=0;i<after.size();++i){auto expected=before[i];for(unsigned a=0;a<3;++a)expected.velocityInvMass[a]=after[i].velocityInvMass[a];require(std::memcmp(&expected,&after[i],64)==0,"Impulse adoption altered nonvelocity state");if(after[i].identity[1]){double square=0;for(unsigned a=0;a<3;++a)square+=double(after[i].velocityInvMass[a])*after[i].velocityInvMass[a];kinetic+=.5*square/after[i].velocityInvMass[3];}}
        require(kinetic>0 && kinetic<=charge && particles.poolSnapshot().active==2 && particles.settlingSnapshot().valid && particles.settlingSnapshot().free==0,"Impulse energy/count/wake failed");
        const auto resetHistory=read(particles,true);const std::array<std::uint32_t,4> zero{};
        for(const auto& p:resetHistory)require(std::memcmp(p.positionRadius,zero.data(),16)==0,"Accepted impulse retained stale settling history");
        for(const auto& mass:particles.debrisMassTotals())require(mass==DebrisMassTotals{},"Impulse changed conversion mass history");
        particles.tick(1,live->occupancy(),live->source());particles.request(6);const auto cleared=read(particles);particles.poll(bgfx::frame());require(particles.pendingCommands()==0 && particles.ticks()==1 && particles.poolSnapshot().active==0 && !particles.fault(),"Deferred clear failed after impulse");for(const auto& p:cleared)require(p.identity[1]==0,"Queued clear did not retire impulse recipients");++cases;
    }
    for(unsigned fault=0;fault<2;++fault) {
        auto live=world(),original=live;VoxelProbes particles(assets,materials,true,true);auto before=populate(particles);
        if(fault==0){before[0].identity[3]=0;bgfx::update(particles.state(),0,bgfx::copy(before.data(),unsigned(before.size()*64)));}
        else {const std::array<std::uint32_t,4> health{1,0,0,0};bgfx::update(particles.status(),0,bgfx::copy(health.data(),16));}
        PendingBlastImpulse pending(assets,particles,live,materials,request);pending.submit(8,particles);wait(pending);
        require(pending.state()==State::Rejected && (pending.errors()&(fault==0?1u:32u)) && pending.energyJoules()==0 && !particles.transactionPending() && !pending.publish(0,particles,live) && live==original,"Faulted impulse was adopted or charged");const auto after=read(particles);require(std::memcmp(before.data(),after.data(),before.size()*64)==0,"Rejected impulse changed live bytes");++cases;
    }
    for(unsigned fault=0;fault<2;++fault) {
        auto live=world();VoxelProbes particles(assets,materials,true,true);const auto before=populate(particles);
        PendingBlastImpulse pending(assets,particles,live,materials,request);pending.submit(8,particles);wait(pending);require(pending.state()==State::Ready,"Stale fixture did not prepare");
        if(fault==0){live=world();require(!pending.publish(0,particles,live),"Same-revision replacement terrain accepted impulse");}
        else {VoxelProbes replacement(assets,materials,true,true);require(!pending.publish(0,replacement,live),"Replacement pool accepted impulse");}
        require(pending.state()==State::Stale && pending.energyJoules()==0 && !particles.transactionPending(),"Stale impulse retained charge/freeze");const auto after=read(particles);require(std::memcmp(before.data(),after.data(),before.size()*64)==0,"Stale impulse changed original pool");++cases;
    }
    {
        auto live=world();VoxelProbes particles(assets,materials,true,true);const auto before=populate(particles);
        {PendingBlastImpulse abandoned(assets,particles,live,materials,request);}
        {PendingBlastImpulse abandoned(assets,particles,live,materials,request);abandoned.submit(8,particles);}
        auto after=read(particles);require(!particles.transactionPending() && std::memcmp(before.data(),after.data(),before.size()*64)==0,"Abandonment changed state/freeze");
        auto changed=materials;changed.entries[0].gpu.mechanical[0]*=2;rejects([&]{PendingBlastImpulse invalid(assets,particles,live,changed,request);});
        const auto handle=particles.state();const std::array<std::uint32_t,4> history{1,30,0,1};bgfx::update(particles.settlingHistory(),0,bgfx::copy(history.data(),16));
        auto empty=request;empty.joules=0;PendingBlastImpulse noop(assets,particles,live,materials,empty);noop.submit(8,particles);wait(noop);require(noop.publish(0,particles,live) && noop.accepted()==0 && noop.energyJoules()==0,"Zero budget impulse failed no-op commit");after=read(particles);require(std::memcmp(before.data(),after.data(),before.size()*64)==0 && particles.state().idx==handle.idx,"Zero budget changed velocity/ownership");const auto unchangedHistory=read(particles,true);require(std::memcmp(unchangedHistory[0].positionRadius,history.data(),16)==0,"Zero effect reset settling history");++cases;
    }
    std::cout<<"GPU acknowledged blast impulse passed: "<<cases<<" scenarios; freeze, immutable preparation, single adoption, energy, settling wake, queued commands, rollback, stale owners and abandonment\n";
}
}
