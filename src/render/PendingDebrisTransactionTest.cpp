#include "PendingDebrisTransaction.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
namespace pw {
namespace {
template<class T>struct TransactionHandle {
    T value;explicit TransactionHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Transaction test allocation failed");}
    ~TransactionHandle(){bgfx::destroy(value);}TransactionHandle(const TransactionHandle&)=delete;
};
}
void runPendingDebrisTransactionTest(const std::filesystem::path& assets) {
    using Store=terrain::SparseVoxelStore;using State=PendingDebrisTransaction::State;
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    auto rejects=[&](auto operation){bool bad=false;try{operation();}catch(const std::invalid_argument&){bad=true;}require(bad,"Invalid debris transaction accepted");};
    const auto materials=loadMaterials(assets/"materials/default.json");
    const auto first=*terrain::addressOf({8,8,8}),second=*terrain::addressOf({10,8,8}),third=*terrain::addressOf({12,8,8});
    const std::array initial{Store::VoxelEdit{first,0},Store::VoxelEdit{second,1},Store::VoxelEdit{third,0}};
    const std::array remove{Store::VoxelEdit{first,Store::empty}};
    auto makeBase=[&]{auto store=std::make_unique<Store>(terrain::ResidentChunkLayout({0,0,0},{1,1,1}),unsigned(materials.entries.size()));store->setMaterials(initial,0);return VoxelPublicationPtr(std::make_shared<VoxelPublication>(assets,std::move(store),terrain::Chunk{},materials));};
    auto seedsFor=[](unsigned material=0,float x=1.0625f){std::vector<DebrisSeed> seeds(2);for(unsigned i=0;i<2;++i){seeds[i].position={x-float(i)*.125f,1.0625f,1.0625f};seeds[i].material=material;}return seeds;};
    TransactionHandle exporter(loadCompute(assets,"cs_export"));TransactionHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    TransactionHandle image(bgfx::createTexture2D(4,VoxelProbes::poolCapacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
    TransactionHandle staging(bgfx::createTexture2D(4,VoxelProbes::poolCapacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
    auto read=[&](VoxelProbes& particles) {
        const float p[]{0,float(particles.capacity()),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,particles.state(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,exporter.value,dispatchGroups(particles.capacity()));
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(7,dst,src);std::vector<ParticleGpu> result(particles.capacity());
        const auto ready=bgfx::read(dst,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
    };
    auto wait=[&](PendingDebrisTransaction& transaction){for(unsigned i=0;i<128 && transaction.state()==State::Waiting;++i){transaction.poll(bgfx::frame());SDL_PumpEvents();}require(transaction.state()!=State::Waiting,"Debris transaction did not acknowledge");};
    unsigned cases=0;
    {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);const auto before=read(particles);
        auto seeds=seedsFor();auto other=seedsFor(1,1.3125f);seeds.insert(seeds.end(),other.begin(),other.end());
        const std::array removals{remove[0],Store::VoxelEdit{second,Store::empty}};
        PendingDebrisTransaction transaction(assets,particles,live,removals,seeds,materials);
        require(particles.transactionPending() && !transaction.publish(1,particles,live),"Unsubmitted transaction published");
        rejects([&]{PendingDebrisTransaction duplicate(assets,particles,live,remove,seedsFor(),materials);});
        rejects([&]{transaction.submit(UINT16_MAX,particles);});require(transaction.state()==State::Prepared,"Invalid view consumed transaction");
        Command command;command.type=CommandType::RemoveBrush;command.minimum={-16,0,-16};command.maximum={16,16,16};
        for(unsigned i=0;i<materials.entries.size();++i)if(materials.entries[i].name=="Sand")command.material=i;
        require(particles.enqueue(command).empty(),"Frozen pool rejected queued command");
        for(unsigned i=0;i<3;++i)particles.tick(1,live->occupancy(),live->source());
        require(particles.ticks()==0 && particles.pendingCommands()==1,"Transaction did not freeze ticks/commands");
        transaction.submit(2,particles);require(!transaction.publish(3,particles,live) && live==original,"Unacknowledged conversion published");
        bool repeated=false;try{transaction.submit(2,particles);}catch(const std::logic_error&){repeated=true;}require(repeated,"Repeated transaction submission accepted");
        wait(transaction);require(transaction.state()==State::Ready && particles.transactionPending(),"Valid conversion rejected");
        const auto staged=read(particles);require(std::memcmp(staged.data(),before.data(),before.size()*64)==0 && live==original,"Preparation changed live terrain/particles");
        rejects([&]{transaction.publish(UINT16_MAX,particles,live);});
        require(transaction.publish(1,particles,live) && !particles.transactionPending() && !transaction.publish(1,particles,live),"Joint conversion failed or repeated");
        require(live!=original && live->source().revision()==2 && live->source().occupiedVoxelCount()==1 && live->mesh().faceCount()==6 && original->source().occupiedVoxelCount()==3,"Mixed terrain revision after conversion");
        particles.request(6);auto activated=read(particles);particles.poll(bgfx::frame());
        require(particles.poolSnapshot().valid && particles.poolSnapshot().active==4 && particles.settlingSnapshot().valid && particles.settlingSnapshot().free==0 && !particles.fault(),"Commit did not rebuild membership/reset settling");unsigned active=0;
        for(unsigned i=0;i<activated.size();++i)if(activated[i].identity[1]) {
            ++active;require(i<4 && activated[i].identity[1]==1 && activated[i].identity[2]==i && activated[i].identity[3]==1 && activated[i].identity[0]==seeds[i].material,"Wrong activated identity");
            require(std::memcmp(activated[i].positionRadius,seeds[i].position.data(),12)==0,"Wrong debris placement");
            require(activated[i].velocityInvMass[3]==terrain::conversionMassProfile(materials.entries[seeds[i].material].gpu.mechanical[0]).inverseMass,"Wrong committed debris mass");
        }
        require(active==4,"Joint conversion lost/duplicated grains");
        for(unsigned material:{0u,1u}) {const auto expected=terrain::planVoxelDebris(1,materials.entries[material].gpu.mechanical[0]);const auto& total=particles.debrisMassTotals()[material];require(total.removedVoxels==1 && total.activatedGrains==2 && total.sourceKilograms==expected.sourceKilograms && total.destinationKilograms==expected.destinationKilograms && total.quantizationKilograms==expected.quantizationKilograms,"Committed per-material ledger mismatch");}
        // Queued Sand removal runs only after joint publication. Conversion counters
        // remain a history of converted mass, not current active-particle mass.
        particles.tick(1,live->occupancy(),live->source());particles.request(6);activated=read(particles);particles.poll(bgfx::frame());require(!particles.fault() && particles.poolSnapshot().active==2,"Physics or membership failed after commit");for(unsigned i=0;i<activated.size();++i)require(activated[i].identity[1]==(i<2?1u:0u),"Deferred material brush did not consume new publication");require(particles.ticks()==1 && particles.pendingCommands()==0,"Deferred tick failed");
        const std::array last{Store::VoxelEdit{third,Store::empty}};PendingDebrisTransaction next(assets,particles,live,last,seedsFor(0,1.5625f),materials);next.submit(2,particles);wait(next);require(next.publish(1,particles,live),"Second conversion failed");
        activated=read(particles);require(activated[2].identity[3]==2 && particles.debrisMassTotals()[0].removedVoxels==2 && particles.debrisMassTotals()[0].activatedGrains==4 && live->source().occupiedVoxelCount()==0,"Generation or cumulative ledger failed");
        particles.tick(1,live->occupancy(),live->source());particles.request(6);read(particles);particles.poll(bgfx::frame());require(!particles.fault() && particles.poolSnapshot().active==4,"Second adopted buffer failed physics/membership");++cases;
    }
    for(unsigned fault=0;fault<5;++fault) {
        auto live=makeBase(),original=live;VoxelProbes particles(assets,materials,true,true);auto before=read(particles);auto seeds=seedsFor();
        if(fault==0)seeds[1]=seeds[0]; // New-new overlap.
        if(fault==1){seeds[0].position={1.3125f,1.0625f,1.0625f};seeds[1].position={1.1875f,1.0625f,1.0625f};} // Remaining terrain.
        if(fault==2 || fault==3) {
            const unsigned count=fault==2?particles.capacity():1;
            for(unsigned i=0;i<count;++i)before[i]={{1.0625f,1.0625f,1.0625f,.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,i,1}};
            bgfx::update(particles.state(),0,bgfx::copy(before.data(),unsigned(before.size()*64)));
        }
        if(fault==4){const std::array<std::uint32_t,4> error{1,0,0,0};bgfx::update(particles.status(),0,bgfx::copy(error.data(),16));}
        PendingDebrisTransaction transaction(assets,particles,live,remove,seeds,materials);transaction.submit(2,particles);wait(transaction);
        const unsigned expectedBit=fault==0 || fault==3?8u:fault==1?16u:fault==2?1u:32u;
        require(transaction.state()==State::Rejected && (transaction.errors()&expectedBit)!=0 && !particles.transactionPending() && !transaction.publish(1,particles,live),"Failed transaction published or retained freeze");
        const auto after=read(particles);require(live==original && std::memcmp(after.data(),before.data(),before.size()*64)==0,"Rejected conversion mutated live state");
        for(const auto& mass:particles.debrisMassTotals())require(mass==DebrisMassTotals{},"Rejected conversion counted mass");
        if(fault<2) {
            PendingDebrisTransaction retry(assets,particles,live,remove,seedsFor(),materials);retry.submit(2,particles);wait(retry);
            require(retry.publish(1,particles,live) && particles.debrisMassTotals()[0].removedVoxels==1,"Rejected batch poisoned retry or double-counted mass");read(particles);
        }
        ++cases;
    }
    for(unsigned fault=0;fault<2;++fault) {
        auto live=makeBase();VoxelProbes particles(assets,materials,true,true);read(particles);
        PendingDebrisTransaction transaction(assets,particles,live,remove,seedsFor(),materials);transaction.submit(2,particles);wait(transaction);require(transaction.state()==State::Ready,"Stale fixture did not prepare");
        if(fault==0) {
            live=std::make_shared<VoxelPublication>(assets,live->source().clone(),terrain::Chunk{},materials);const auto replacement=live;
            require(!transaction.publish(1,particles,live) && live==replacement,"Same-revision replacement terrain accepted");
        }else {
            VoxelProbes replacement(assets,materials,true,true);const auto original=live;
            require(!transaction.publish(1,replacement,live) && live==original,"Same-tick replacement pool accepted");
        }
        require(transaction.state()==State::Stale && !particles.transactionPending(),"Stale transaction retained lease");++cases;
    }
    {
        auto live=makeBase();VoxelProbes particles(assets,materials,true,true);const auto before=read(particles);
        {PendingDebrisTransaction abandoned(assets,particles,live,remove,seedsFor(),materials);}
        require(!particles.transactionPending(),"Abandoned prepared transaction retained lease");
        {PendingDebrisTransaction abandoned(assets,particles,live,remove,seedsFor(),materials);abandoned.submit(2,particles);}
        const auto after=read(particles);require(!particles.transactionPending() && std::memcmp(before.data(),after.data(),before.size()*64)==0 && live->source().occupiedVoxelCount()==3,"Abandoned in-flight transaction mutated authority");
        auto shortSeeds=seedsFor();shortSeeds.pop_back();rejects([&]{PendingDebrisTransaction t(assets,particles,live,remove,shortSeeds,materials);});
        auto wrongSeeds=seedsFor(1);rejects([&]{PendingDebrisTransaction t(assets,particles,live,remove,wrongSeeds,materials);});
        rejects([&]{PendingDebrisTransaction t(assets,particles,live,remove,seedsFor(),materials,0);});
        auto changed=materials;changed.contentHash^=1;rejects([&]{PendingDebrisTransaction t(assets,particles,live,remove,seedsFor(),changed);});
        changed=materials;changed.entries[0].gpu.mechanical[0]*=2;rejects([&]{PendingDebrisTransaction t(assets,particles,live,remove,seedsFor(),changed);});
        const std::array duplicate{remove[0],remove[0]};auto many=seedsFor();auto more=seedsFor(0,2);many.insert(many.end(),more.begin(),more.end());rejects([&]{PendingDebrisTransaction t(assets,particles,live,duplicate,many,materials);});
        const std::array addition{Store::VoxelEdit{first,1}};rejects([&]{PendingDebrisTransaction t(assets,particles,live,addition,seedsFor(),materials);});
        const std::array empty{Store::VoxelEdit{*terrain::addressOf({0,0,0}),Store::empty}};rejects([&]{PendingDebrisTransaction t(assets,particles,live,empty,seedsFor(),materials);});
        rejects([&]{PendingDebrisTransaction t(assets,particles,live,{}, {},materials);});
        require(!particles.transactionPending(),"Failed construction froze pool");++cases;
    }
    {
        auto live=makeBase();auto old=std::make_unique<VoxelProbes>(assets,materials,true,true);read(*old);
        PendingDebrisTransaction transaction(assets,*old,live,remove,seedsFor(),materials);transaction.submit(2,*old);old.reset();
        wait(transaction);VoxelProbes replacement(assets,materials,true,true);const auto before=read(replacement);const auto original=live;
        require(!transaction.publish(1,replacement,live) && transaction.state()==State::Stale && live==original && !replacement.transactionPending(),"Destroyed particle owner was resurrected");
        const auto after=read(replacement);require(std::memcmp(before.data(),after.data(),before.size()*64)==0,"Old transaction changed replacement particles");++cases;
    }
    std::cout<<"GPU joint debris transaction passed: "<<cases<<" scenarios; frozen ticks, dual acknowledgement, atomic terrain/particle/mass publication, failures, stale owners, generations and abandonment\n";
}
}
