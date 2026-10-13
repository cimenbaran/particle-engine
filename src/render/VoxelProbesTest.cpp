#include "VoxelProbes.hpp"
#include "Shaders.hpp"
#include "core/BasinWorld.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
#include <algorithm>
namespace pw {
static void voxelProbesCase(const std::filesystem::path& assets,bool interacting,bool pooled=false,bool brushSoak=false) {
    auto require=[](bool good,const char* message){if(!good)throw std::runtime_error(message);};
    const auto materials=loadMaterials(assets/"materials/default.json");auto store=terrain::buildBasin(1337,std::uint32_t(materials.entries.size()),0,1,3);GpuVoxelWorld world(*store,{});
    VoxelProbes probes(assets,materials,interacting,pooled);const auto capacity=probes.capacity();auto frame=bgfx::frame();
    struct Handles {
        bgfx::ProgramHandle program=BGFX_INVALID_HANDLE;bgfx::UniformHandle uniform=BGFX_INVALID_HANDLE;
        bgfx::TextureHandle image=BGFX_INVALID_HANDLE,staging=BGFX_INVALID_HANDLE;
        ~Handles(){if(bgfx::isValid(program))bgfx::destroy(program);if(bgfx::isValid(uniform))bgfx::destroy(uniform);for(auto h:{image,staging})if(bgfx::isValid(h))bgfx::destroy(h);}
    } h;
    h.program=loadCompute(assets,"cs_export");h.uniform=bgfx::createUniform("u_sim",bgfx::UniformType::Vec4);
    h.image=bgfx::createTexture2D(4,capacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE);
    h.staging=bgfx::createTexture2D(4,capacity,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    require(bgfx::isValid(h.image) && bgfx::isValid(h.staging) && bgfx::isValid(h.uniform),"Probe test export allocation failed");
    auto snapshot=[&]() {
        const float p[]{0,float(capacity),0,4};bgfx::setUniform(h.uniform,p);bgfx::setBuffer(0,probes.state(),bgfx::Access::Read);bgfx::setImage(1,h.image,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,h.program,dispatchGroups(capacity));
        bgfx::TextureRegion src,dst;src.handle=h.image;dst.handle=h.staging;bgfx::blit(7,dst,src);std::vector<ParticleGpu> result(capacity);const auto ready=bgfx::read(dst,result.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();probes.poll(frame);return result;
    };
    auto historySnapshot=[&]() {
        std::vector<SettlingState> result(capacity);
        if(!interacting)return result;
        const float p[]{0,float(capacity),0,1};bgfx::setUniform(h.uniform,p);bgfx::setBuffer(0,probes.settlingHistory(),bgfx::Access::Read);bgfx::setImage(1,h.image,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,h.program,dispatchGroups(capacity));
        bgfx::TextureRegion src,dst;src.handle=h.image;dst.handle=h.staging;bgfx::blit(7,dst,src);
        probes.request(6);
        std::vector<std::array<std::uint32_t,16>> rows(capacity);
        const auto ready=bgfx::read(dst,rows.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();probes.poll(frame);
        for(unsigned i=0;i<result.size();++i)std::memcpy(&result[i],rows[i].data(),sizeof(SettlingState));
        return result;
    };
    if(brushSoak) {
        Command spawn;spawn.type=CommandType::SpawnBrush;spawn.material=1;spawn.count=256;spawn.minimum={-6.5f,5.5f,-5};spawn.maximum={-5.5f,6.5f,-4};spawn.parameters={2,0,0,0};
        require(probes.enqueue(spawn).empty(),"Default basin brush enqueue failed");
        for(unsigned tick=0;tick<600;++tick){probes.tick(1,world,*store);probes.request(6);frame=bgfx::frame();probes.poll(frame);if(probes.fault())throw std::runtime_error("Default basin brush fault tick "+std::to_string(tick)+" phases "+std::to_string(probes.errors()));}
        const auto final=snapshot();require(!probes.fault(),"Default brush final fault");
        unsigned live=0;
        for(unsigned i=0;i<capacity;++i)if(final[i].identity[1]&1u) {
            ++live;const auto& p=final[i];require(p.identity[0]==1 && p.identity[2]==i && p.identity[3]==1,"Default brush identity changed");
            require(std::abs(p.thermalState[3]-10.0f)<.001f && std::hypot(p.velocityInvMass[0],p.velocityInvMass[1],p.velocityInvMass[2])<=12.00002f,"Default brush age/speed invalid");
            reference::VoxelVector position{p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]};
            terrain::GlobalVoxel centre{};for(unsigned a=0;a<3;++a){require(std::isfinite(position[a]),"Default brush position is nonfinite");centre[a]=std::int64_t(std::floor(position[a]*8));}
            for(int z=-1;z<=1;++z)for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
                auto a=terrain::addressOf({centre[0]+x,centre[1]+y,centre[2]+z});if(!a)continue;auto m=store->materialAt(*a);
                if(m && *m!=terrain::SparseVoxelStore::empty)require(reference::voxelDistance(position,*terrain::relativeCorner(*a,{}))>=.0625-1e-7,"Default brush penetrated terrain");
            }
        }
        require(live==256 && probes.poolSnapshot().valid && probes.poolSnapshot().active==live,"Default brush lost particles");
        std::cout<<"GPU default basin brush passed: 600 ticks\n";return;
    }
    if(pooled) {
        auto checkCounts=[&](unsigned active,unsigned rejected) {
            probes.request(6);const auto particles=snapshot();
            const auto& pool=probes.poolSnapshot();
            require(pool.valid && pool.active==active && pool.free==capacity-active && pool.rejected==rejected,"Live basin pool counts mismatch");
            unsigned live=0;
            for(unsigned i=0;i<capacity;++i)if(particles[i].identity[1]&1u){++live;require(particles[i].identity[2]==i && particles[i].identity[3]>0,"Live pool identity mismatch");}
            require(live==active && !probes.fault(),"Live pool state/count or physics mismatch");
            return particles;
        };
        auto advance=[&](){probes.tick(1,world,*store);probes.prepare(5);};
        Command spawn;spawn.type=CommandType::SpawnBrush;spawn.material=1;spawn.count=1;spawn.minimum=spawn.maximum={-6,6,-5};
        Command clear=spawn;clear.type=CommandType::RemoveBrush;clear.count=0;clear.minimum={-16,-16,-16};clear.maximum={16,16,16};
        checkCounts(0,0);
        require(probes.enqueue(spawn).empty() && probes.enqueue(spawn).empty(),"Live pool rejected valid spawn");
        checkCounts(0,0);require(probes.ticks()==0 && probes.pendingCommands()==2,"Queued commands executed before tick");
        advance();checkCounts(1,1);require(probes.pendingCommands()==0 && probes.ticks()==1,"Live command tick mismatch");
        require(probes.enqueue(clear).empty() && probes.enqueue(spawn).empty() && probes.enqueue(clear).empty(),"Live ordered command enqueue failed");
        advance();checkCounts(0,1);
        const auto emptyHistory=historySnapshot();for(const auto& h:emptyHistory)require(!h.quietTicks && !h.candidate && !h.unsupportedTicks,"Removed grains retained settling history");
        spawn.count=256;spawn.maximum={-5.125f,6.375f,-4.125f};
        Command impulse=clear;impulse.type=CommandType::ApplyImpulse;impulse.parameters={1,0,0,0};
        require(probes.enqueue(spawn).empty() && probes.enqueue(impulse).empty(),"Live refill/impulse enqueue failed");
        advance();const auto moving=checkCounts(256,1);
        for(const auto& p:moving)if(p.identity[1]&1u)require(p.velocityInvMass[0]>.95f && p.velocityInvMass[0]<=1.00001f,"Impulse did not follow spawn in command order");
        require(probes.enqueue(clear).empty(),"Live clear enqueue failed");advance();checkCounts(0,1);
        spawn.count=1;spawn.maximum=spawn.minimum;spawn.parameters[3]=.02f;
        require(probes.enqueue(spawn).empty(),"Live lifetime enqueue failed");advance();checkCounts(1,1);advance();checkCounts(0,1);
        Command invalid=spawn;invalid.type=CommandType::ApplyHeat;require(!probes.enqueue(invalid).empty(),"Unsupported live command accepted");
        invalid=spawn;invalid.parameters[0]=13;require(!probes.enqueue(invalid).empty(),"Unbounded live brush velocity accepted");
        spawn.parameters[3]=0;spawn.count=4096;
        require(probes.enqueue(spawn).empty() && probes.enqueue(spawn).empty() && !probes.enqueue(spawn).empty(),"Live tick spawn quota not enforced");
        // Stale publication must not consume queued commands or advance the tick.
        const auto oldTick=probes.ticks();store->setMaterial(*terrain::addressOf({0,4,0}),0,store->revision());
        bool stale=false;try{advance();}catch(const std::invalid_argument&){stale=true;}
        require(stale && probes.ticks()==oldTick && probes.pendingCommands()==2,"Stale terrain consumed live commands");
        GpuVoxelWorld fresh(*store,{});probes.tick(1,fresh,*store);checkCounts(1,8192);
        spawn.count=1024;spawn.maximum={-4.125f,6.375f,-3.125f};
        require(probes.enqueue(clear).empty() && probes.enqueue(spawn).empty(),"Full pool enqueue failed");
        probes.tick(1,fresh,*store);const auto full=checkCounts(1024,8192);
        require(probes.enqueue(clear).empty() && probes.enqueue(spawn).empty(),"Full pool recycle enqueue failed");
        probes.tick(1,fresh,*store);const auto recycled=checkCounts(1024,8192);
        for(unsigned i=0;i<capacity;++i)require(recycled[i].identity[3]==full[i].identity[3]+1,"Live pool reused a stale generation");
        const auto history=historySnapshot();for(unsigned i=0;i<capacity;++i)require(history[i].generation==recycled[i].identity[3] && history[i].quietTicks<=1 && !history[i].candidate,"Recycled slot inherited settling state");
        std::cout<<"GPU live basin command pool passed: empty/queued state, ordered spawn/remove/impulse, duplicate rejection, membership, expiry, settling reset, quotas and stale publication\n";
        return;
    }
    const auto initial=snapshot();
    unsigned peakContacts=0;
    for(unsigned tick=0;tick<600;++tick) {
        probes.tick(1,world,*store);probes.prepare(5);probes.request(6);frame=bgfx::frame();probes.poll(frame);
        peakContacts=std::max(peakContacts,probes.contactDegree());
        if(probes.fault())throw std::runtime_error(std::string(interacting?"Interacting basin":"Live voxel probe")+" fault at tick "+std::to_string(tick)+" errors "+std::to_string(probes.errors()));
    }
    const auto actual=snapshot();require(!probes.fault(),"Final voxel probe fault");
    const auto settled=historySnapshot();
    if(interacting) {
        const auto& counts=probes.settlingSnapshot();require(counts.valid && counts.active==0,"Live basin settling fault");
        unsigned candidates=0,acquiring=0,unsupported=0;
        for(unsigned i=0;i<settled.size();++i) {
            const auto& history=settled[i];require(history.generation==actual[i].identity[3],"Basin settling generation mismatch");
            require(history.candidate<=1 && history.quietTicks<=30 && history.unsupportedTicks<3,"Invalid live settling history");
            candidates+=history.candidate;acquiring+=!history.candidate && history.quietTicks>0;unsupported+=history.candidate && history.unsupportedTicks>0;
            if(history.candidate)require(std::hypot(actual[i].velocityInvMass[0],actual[i].velocityInvMass[1],actual[i].velocityInvMass[2])<=.100001f,"Fast basin particle retained settling candidate");
        }
        require(counts.free==candidates && counts.reserved==acquiring && counts.rejected==unsupported,"Live basin settling diagnostics mismatch");
        std::cout<<"Live basin settling: "<<candidates<<" candidates, "<<acquiring<<" acquiring, "<<unsupported<<" in support grace\n";
    }
    unsigned belowLedge=0;
    for(unsigned id=0;id<actual.size();++id) {
        const auto& p=actual[id];require(p.positionRadius[3]==initial[id].positionRadius[3] && p.velocityInvMass[3]==initial[id].velocityInvMass[3],"Basin particle radius or mass changed");require(p.identity[0]==1 && p.identity[1]==1 && p.identity[2]==id && p.identity[3]==1,"Probe identity changed");
        require(p.positionRadius[1]<(interacting?5.0f:4.3f) && p.positionRadius[1]>=.5625f && std::abs(p.thermalState[3]-10.0f)<.001,"Probe did not advance under gravity");
        if(p.positionRadius[1]<3.8f)++belowLedge;
        reference::VoxelVector position{p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]};
        terrain::GlobalVoxel lower{},upper{};
        for(unsigned axis=0;axis<3;++axis){require(std::isfinite(position[axis]) && std::isfinite(p.velocityInvMass[axis]),"Nonfinite live probe");lower[axis]=std::int64_t(std::floor((position[axis]-.0625)*8))-1;upper[axis]=std::int64_t(std::ceil((position[axis]+.0625)*8))+1;}
        for(auto z=lower[2];z<=upper[2];++z)for(auto y=lower[1];y<=upper[1];++y)for(auto x=lower[0];x<=upper[0];++x) {
            auto address=terrain::addressOf({x,y,z});if(!address)continue;const auto material=store->materialAt(*address);
            if(material && *material!=terrain::SparseVoxelStore::empty)
                require(reference::voxelDistance(position,*terrain::relativeCorner(*address,{}))>=.0625-1e-7,"Live probe penetrated basin terrain");
        }
    }
    if(interacting) {
        unsigned contacts=0,deep=0;double maxPenetration=0,maxSpeed=0;
        for(unsigned i=0;i<actual.size();++i) {
            maxSpeed=std::max(maxSpeed,std::hypot(double(actual[i].velocityInvMass[0]),actual[i].velocityInvMass[1],actual[i].velocityInvMass[2]));
            for(unsigned j=0;j<i;++j){double squared=0;for(unsigned a=0;a<3;++a){const double d=double(actual[i].positionRadius[a])-actual[j].positionRadius[a];squared+=d*d;}const double overlap=.125-std::sqrt(squared);if(overlap>0){++contacts;if(overlap>.00625)++deep;maxPenetration=std::max(maxPenetration,overlap);}}
        }
        require(deep*100<=std::max(1u,contacts),"Basin has too many deep particle overlaps");require(maxSpeed<=12.00002,"Basin speed limit exceeded");
        require(maxSpeed<=.05,"Terrain traction failed the 10-second low-speed fixture");
        std::cout<<"Interacting basin final: peak degree "<<peakContacts<<", max penetration "<<maxPenetration<<", deep "<<deep<<"/"<<contacts<<", max speed "<<maxSpeed<<"\n";
    }
    if(interacting)require(peakContacts>0,"Interacting basin did not execute particle contacts");
    require(belowLedge>=16,"Live probes did not exercise the lower slope");
    // An occupancy edit invalidates the snapshot before any force/age work.
    store->setMaterial(*terrain::addressOf({0,4,0}),0,store->revision());bool stale=false;
    try{probes.tick(1,world,*store);}catch(const std::invalid_argument&){stale=true;}
    require(stale && probes.ticks()==600,"Stale live probe tick was submitted");const auto unchanged=snapshot();require(std::memcmp(actual.data(),unchanged.data(),actual.size()*sizeof(ParticleGpu))==0,"Stale probe rejection mutated state");
    if(interacting){const auto afterStale=historySnapshot();require(afterStale==settled,"Stale basin tick advanced settling history");}
    const auto& occupiedProbe=actual[0];
    store->setMaterial(*terrain::addressAtRelative({occupiedProbe.positionRadius[0],occupiedProbe.positionRadius[1],occupiedProbe.positionRadius[2]}),0,store->revision());
    GpuVoxelWorld overlapWorld(*store,{});probes.tick(1,overlapWorld,*store);probes.request(6);
    const auto rolledBack=snapshot();require(probes.fault(),"Overlapping terrain did not latch a probe fault");
    require(std::memcmp(actual.data(),rolledBack.data(),actual.size()*sizeof(ParticleGpu))==0,"Probe failure did not roll back the entire tick");
    if(interacting) {
        const auto failed=historySnapshot();for(const auto& history:failed)require(history.quietTicks==0 && history.candidate==0 && history.unsupportedTicks==0,"Physics rollback retained basin settling history");
        require(probes.settlingSnapshot().active!=0,"Physics rollback did not invalidate settling diagnostics");
    }
    std::cout<<(interacting?"GPU interacting basin":"GPU live voxel probes")<<" passed: 256 particles, 600 ticks / 2400 substeps, gravity/drag, full basin containment, identity, stale-world rejection and whole-tick rollback\n";
}
void runVoxelProbesTest(const std::filesystem::path& assets){voxelProbesCase(assets,false);voxelProbesCase(assets,true);voxelProbesCase(assets,true,true);voxelProbesCase(assets,true,true,true);}

}
