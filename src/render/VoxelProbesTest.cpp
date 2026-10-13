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
static void voxelProbesCase(const std::filesystem::path& assets,bool interacting) {
    auto require=[](bool good,const char* message){if(!good)throw std::runtime_error(message);};
    const auto materials=loadMaterials(assets/"materials/default.json");auto store=terrain::buildBasin(1337,std::uint32_t(materials.entries.size()),0,1,3);GpuVoxelWorld world(*store,{});
    VoxelProbes probes(assets,materials,interacting);auto frame=bgfx::frame();
    struct Handles {
        bgfx::ProgramHandle program=BGFX_INVALID_HANDLE;bgfx::UniformHandle uniform=BGFX_INVALID_HANDLE;
        bgfx::TextureHandle image=BGFX_INVALID_HANDLE,staging=BGFX_INVALID_HANDLE;
        ~Handles(){if(bgfx::isValid(program))bgfx::destroy(program);if(bgfx::isValid(uniform))bgfx::destroy(uniform);for(auto h:{image,staging})if(bgfx::isValid(h))bgfx::destroy(h);}
    } h;
    h.program=loadCompute(assets,"cs_export");h.uniform=bgfx::createUniform("u_sim",bgfx::UniformType::Vec4);
    h.image=bgfx::createTexture2D(4,VoxelProbes::count,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE);
    h.staging=bgfx::createTexture2D(4,VoxelProbes::count,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    require(bgfx::isValid(h.image) && bgfx::isValid(h.staging) && bgfx::isValid(h.uniform),"Probe test export allocation failed");
    auto snapshot=[&]() {
        const float p[]{0,float(VoxelProbes::count),0,4};bgfx::setUniform(h.uniform,p);bgfx::setBuffer(0,probes.state(),bgfx::Access::Read);bgfx::setImage(1,h.image,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,h.program,dispatchGroups(VoxelProbes::count));
        bgfx::TextureRegion src,dst;src.handle=h.image;dst.handle=h.staging;bgfx::blit(7,dst,src);std::array<ParticleGpu,VoxelProbes::count> result{};const auto ready=bgfx::read(dst,result.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();probes.poll(frame);return result;
    };
    auto historySnapshot=[&]() {
        std::array<SettlingState,VoxelProbes::count> result{};
        if(!interacting)return result;
        const float p[]{0,float(VoxelProbes::count),0,1};bgfx::setUniform(h.uniform,p);bgfx::setBuffer(0,probes.settlingHistory(),bgfx::Access::Read);bgfx::setImage(1,h.image,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,h.program,dispatchGroups(VoxelProbes::count));
        bgfx::TextureRegion src,dst;src.handle=h.image;dst.handle=h.staging;bgfx::blit(7,dst,src);
        probes.request(6);
        std::array<std::array<std::uint32_t,16>,VoxelProbes::count> rows{};
        const auto ready=bgfx::read(dst,rows.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();probes.poll(frame);
        for(unsigned i=0;i<result.size();++i)std::memcpy(&result[i],rows[i].data(),sizeof(SettlingState));
        return result;
    };
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
        std::cout<<"Interacting basin final: peak degree "<<peakContacts<<", max penetration "<<maxPenetration<<", deep "<<deep<<"/"<<contacts<<", max speed "<<maxSpeed<<"\n";
    }
    if(interacting)require(peakContacts>0,"Interacting basin did not execute particle contacts");
    require(belowLedge>=16,"Live probes did not exercise the lower slope");
    // An occupancy edit invalidates the snapshot before any force/age work.
    store->setMaterial(*terrain::addressOf({0,4,0}),0,store->revision());bool stale=false;
    try{probes.tick(1,world,*store);}catch(const std::invalid_argument&){stale=true;}
    require(stale && probes.ticks()==600,"Stale live probe tick was submitted");const auto unchanged=snapshot();require(std::memcmp(actual.data(),unchanged.data(),sizeof(actual))==0,"Stale probe rejection mutated state");
    if(interacting){const auto afterStale=historySnapshot();require(afterStale==settled,"Stale basin tick advanced settling history");}
    const auto& occupiedProbe=actual[0];
    store->setMaterial(*terrain::addressAtRelative({occupiedProbe.positionRadius[0],occupiedProbe.positionRadius[1],occupiedProbe.positionRadius[2]}),0,store->revision());
    GpuVoxelWorld overlapWorld(*store,{});probes.tick(1,overlapWorld,*store);probes.request(6);
    const auto rolledBack=snapshot();require(probes.fault(),"Overlapping terrain did not latch a probe fault");
    require(std::memcmp(actual.data(),rolledBack.data(),sizeof(actual))==0,"Probe failure did not roll back the entire tick");
    if(interacting) {
        const auto failed=historySnapshot();for(const auto& history:failed)require(history.quietTicks==0 && history.candidate==0 && history.unsupportedTicks==0,"Physics rollback retained basin settling history");
        require(probes.settlingSnapshot().active!=0,"Physics rollback did not invalidate settling diagnostics");
    }
    std::cout<<(interacting?"GPU interacting basin":"GPU live voxel probes")<<" passed: 256 particles, 600 ticks / 2400 substeps, gravity/drag, full basin containment, identity, stale-world rejection and whole-tick rollback\n";
}
void runVoxelProbesTest(const std::filesystem::path& assets){voxelProbesCase(assets,false);voxelProbesCase(assets,true);}

}
