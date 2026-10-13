#include "StagedExplosion.hpp"
#include "core/ParticleTypes.hpp"
#include <SDL3/SDL.h>
#include <iostream>
namespace pw {
void runStagedExplosionTest(const std::filesystem::path& assets) {
    using Store=terrain::SparseVoxelStore;using State=StagedExplosion::State;
    const auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    const auto rejects=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad,"Invalid staged blast accepted");};
    const auto materials=loadMaterials(assets/"materials/default.json");
    std::vector<terrain::ExplosionMaterial> profiles;for(const auto& m:materials.entries)profiles.push_back({m.gpu.mechanical[0],m.gpu.identity[1]==0,1000,0});
    const auto makeBase=[&] {
        auto source=std::make_unique<Store>(terrain::ResidentChunkLayout({0,0,0},{1,1,1}),unsigned(materials.entries.size()));
        const std::array edits{Store::VoxelEdit{*terrain::addressOf({8,8,8}),0},Store::VoxelEdit{*terrain::addressOf({10,8,8}),1},Store::VoxelEdit{*terrain::addressOf({12,8,8}),0}};
        source->setMaterials(edits,0);return VoxelPublicationPtr(std::make_shared<VoxelPublication>(assets,std::move(source),terrain::Chunk{},materials));
    };
    terrain::ExplosionRequest request;request.centre={1.3125f,1.0625f,1.0625f};request.radius=.6f;request.impulse=0;request.energyJoules=10000;request.voxelQuota=1;request.grainQuota=2;
    std::uint32_t frame=bgfx::frame();
    const auto advance=[&](VoxelProbes& particles){particles.prepare(5);particles.request(6);frame=bgfx::frame();particles.poll(frame);SDL_PumpEvents();};
    const auto awaitBatch=[&](StagedExplosion& blast,VoxelProbes& particles,VoxelPublicationPtr& live,std::uint64_t count) {
        for(unsigned tries=0;tries<128 && blast.active() && blast.batches()<count;++tries){blast.pump(frame,particles,live);advance(particles);}
        require(blast.batches()==count,"Staged blast batch did not publish");
    };
    unsigned cases=0;
    {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);StagedExplosion blast(assets,particles,live,materials,request,profiles);
        rejects([&]{blast.pump(frame,particles,live,8,8);});
        blast.pump(frame,particles,live);require(blast.state()==State::WaitingForGpu && particles.transactionPending() && live==original && blast.remainingJoules()==request.energyJoules,"Unacknowledged blast changed authority/budget");
        particles.tick(1,live->occupancy(),live->source());require(particles.ticks()==0,"Pending blast did not freeze tick");
        awaitBatch(blast,particles,live,1);require(blast.removedVoxels()==1 && blast.activatedGrains()==2 && live->source().occupiedVoxelCount()==2,"First staged batch counts wrong");
        auto remaining=blast.remainingJoules();const auto accepted=live;
        for(unsigned i=0;i<8;++i){blast.pump(frame,particles,live);advance(particles);}
        require(blast.batches()==1 && blast.remainingJoules()==remaining && live==accepted && !particles.transactionPending(),"Paused blast repeated a commit or prepared at the same tick");
        for(unsigned batch=2;batch<=3;++batch) {
            particles.tick(1,live->occupancy(),live->source());advance(particles);
            awaitBatch(blast,particles,live,batch);require(blast.remainingJoules()<remaining,"Published batch was not charged");remaining=blast.remainingJoules();
        }
        particles.tick(1,live->occupancy(),live->source());advance(particles);blast.pump(frame,particles,live);
        require(blast.state()==State::Complete && blast.completion()==StagedExplosion::Completion::Empty && live->source().occupiedVoxelCount()==0 && blast.removedVoxels()==3 && blast.activatedGrains()==6,"Staged blast did not complete all three batches");
        for(unsigned i=0;i<8;++i)advance(particles);
        require(!particles.fault() && particles.poolSnapshot().valid && particles.poolSnapshot().active==6 && blast.chargedJoules()<=request.energyJoules,"Staged blast lost grains or exceeded its budget");++cases;
    }
    {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);StagedExplosion cancelled(assets,particles,live,materials,request,profiles);
        cancelled.pump(frame,particles,live);cancelled.cancel();
        require(cancelled.state()==State::Cancelled && live==original && cancelled.remainingJoules()==request.energyJoules && !particles.transactionPending(),"Cancelled pending blast committed or retained freeze");
        StagedExplosion partial(assets,particles,live,materials,request,profiles);awaitBatch(partial,particles,live,1);const auto remaining=partial.remainingJoules();const auto accepted=live;
        partial.cancel();particles.tick(1,live->occupancy(),live->source());partial.pump(frame,particles,live);advance(particles);
        require(partial.state()==State::Cancelled && live==accepted && live->source().occupiedVoxelCount()==2 && partial.batches()==1 && partial.remainingJoules()==remaining,"Cancel rolled back a committed batch or spent more energy");++cases;
    }
    {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);advance(particles);
        std::vector<ParticleGpu> full(particles.capacity());
        for(unsigned i=0;i<full.size();++i)full[i]={{3,3,3,.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,i,1}};
        bgfx::update(particles.state(),0,bgfx::copy(full.data(),unsigned(full.size()*sizeof(ParticleGpu))));
        StagedExplosion blast(assets,particles,live,materials,request,profiles);
        for(unsigned i=0;i<128 && blast.state()!=State::Deferred;++i){blast.pump(frame,particles,live);advance(particles);}
        require(blast.state()==State::Deferred && (blast.errors()&1) && !particles.transactionPending() && live==original && blast.remainingJoules()==request.energyJoules,"Full pool did not defer intact/unspent");
        for(unsigned i=0;i<8;++i){blast.pump(frame,particles,live);advance(particles);require(!particles.transactionPending(),"Deferred paused blast spun another transaction");}
        Command clear;clear.type=CommandType::RemoveBrush;clear.material=0;clear.minimum={-16,0,-16};clear.maximum={16,16,16};require(particles.enqueue(clear).empty(),"Failed to clear capacity fixture");
        for(unsigned tick=0;tick<StagedExplosion::retryTicks;++tick) {
            particles.tick(1,live->occupancy(),live->source());advance(particles);
            if(tick+1<StagedExplosion::retryTicks){blast.pump(frame,particles,live);require(blast.state()==State::Deferred && !particles.transactionPending(),"Capacity retry ran before its bounded interval");}
        }
        awaitBatch(blast,particles,live,1);require(blast.remainingJoules()<request.energyJoules && live->source().occupiedVoxelCount()==2,"Recovered capacity did not resume staged blast");blast.cancel();++cases;
    }
    {
        auto live=makeBase();VoxelProbes particles(assets,materials,true,true);StagedExplosion blast(assets,particles,live,materials,request,profiles);
        blast.pump(frame,particles,live);live=makeBase();const auto replacement=live;blast.pump(frame,particles,live);
        require(blast.state()==State::Stale && live==replacement && blast.remainingJoules()==request.energyJoules && !particles.transactionPending(),"Replaced terrain accepted an old staged blast");
        StagedExplosion other(assets,particles,live,materials,request,profiles);VoxelProbes newParticles(assets,materials,true,true);other.pump(frame,newParticles,live);
        require(other.state()==State::Stale && other.remainingJoules()==request.energyJoules,"Replacement pool accepted old blast");
        auto bad=profiles;bad[0].density*=.5f;rejects([&]{StagedExplosion wrong(assets,particles,live,materials,request,bad);});bad=profiles;bad[0].solid=false;rejects([&]{StagedExplosion wrong(assets,particles,live,materials,request,bad);});++cases;
    }
    {
        auto live=makeBase();VoxelProbes particles(assets,materials,true,true);auto zero=request;zero.energyJoules=0;StagedExplosion exhausted(assets,particles,live,materials,zero,profiles);exhausted.pump(frame,particles,live);
        require(exhausted.state()==State::Complete && exhausted.completion()==StagedExplosion::Completion::Energy && exhausted.batches()==0 && !particles.transactionPending(),"Empty energy budget mutated blast");
        zero=request;zero.grainQuota=0;StagedExplosion quota(assets,particles,live,materials,zero,profiles);quota.pump(frame,particles,live);
        require(quota.state()==State::Complete && quota.completion()==StagedExplosion::Completion::Quota && quota.remainingJoules()==request.energyJoules,"Impossible quota spent energy");
        StagedExplosion queued(assets,particles,live,materials,request,profiles);Command command;command.type=CommandType::RemoveBrush;command.minimum={-16,0,-16};command.maximum={16,16,16};require(particles.enqueue(command).empty(),"Queue fixture failed");queued.pump(frame,particles,live);
        require(!particles.transactionPending() && queued.batches()==0 && particles.pendingCommands()==1,"Blast overtook queued particle command");particles.tick(1,live->occupancy(),live->source());advance(particles);awaitBatch(queued,particles,live,1);queued.cancel();++cases;
    }
    {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);advance(particles);
        const std::array<std::uint32_t,4> error{1,0,0,0};bgfx::update(particles.status(),0,bgfx::copy(error.data(),16));
        StagedExplosion blast(assets,particles,live,materials,request,profiles);
        for(unsigned i=0;i<128 && blast.active();++i){blast.pump(frame,particles,live);advance(particles);}
        require(blast.state()==State::Faulted && blast.batches()==0 && blast.remainingJoules()==request.energyJoules && live==original && !particles.transactionPending(),"Faulted blast published or kept retrying");++cases;
    }
    {
        auto source=std::make_unique<Store>(terrain::ResidentChunkLayout({0,0,0},{1,1,1}),unsigned(materials.entries.size()));
        std::vector<Store::VoxelEdit> edits;for(int y=0;y<32;++y)for(int z=0;z<32;++z)for(int x=0;x<32;++x)edits.push_back({*terrain::addressOf({x,y,z}),0});source->setMaterials(edits,0);
        VoxelPublicationPtr live=std::make_shared<VoxelPublication>(assets,std::move(source),terrain::Chunk{},materials);const auto original=live;
        VoxelProbes particles(assets,materials,true,true);auto buried=request;buried.centre={2.0625f,2.0625f,2.0625f};buried.radius=.01f;
        StagedExplosion blast(assets,particles,live,materials,buried,profiles);blast.pump(frame,particles,live);
        require(blast.state()==State::NoSpace && !particles.transactionPending() && blast.batches()==0 && blast.remainingJoules()==request.energyJoules && live==original,"Buried blast did not stop intact and unspent");++cases;
    }
    std::cout<<"GPU staged explosion passed: "<<cases<<" scenarios; multi-tick publication, energy ownership, paused/cancelled/stale commands, full-pool recovery and bounded retry\n";
}
}
