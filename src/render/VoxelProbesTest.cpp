#include "VoxelProbes.hpp"
#include "Shaders.hpp"
#include "core/BasinWorld.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
namespace pw {
void runVoxelProbesTest(const std::filesystem::path& assets) {
    auto require=[](bool good,const char* message){if(!good)throw std::runtime_error(message);};
    const auto materials=loadMaterials(assets/"materials/default.json");auto store=terrain::buildBasin(1337,std::uint32_t(materials.entries.size()),0,1,3);GpuVoxelWorld world(*store,{});
    VoxelProbes probes(assets,materials);auto frame=bgfx::frame();
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
    for(unsigned tick=0;tick<600;++tick) {
        probes.tick(1,world,*store);probes.prepare(5);probes.request(6);frame=bgfx::frame();probes.poll(frame);
        if(probes.fault())throw std::runtime_error("Live voxel probe fault at tick "+std::to_string(tick)+" errors "+std::to_string(probes.errors()));
    }
    const auto actual=snapshot();require(!probes.fault(),"Final voxel probe fault");
    unsigned belowLedge=0;
    for(unsigned id=0;id<actual.size();++id) {
        const auto& p=actual[id];require(p.identity[0]==1 && p.identity[1]==1 && p.identity[2]==id && p.identity[3]==1,"Probe identity changed");
        require(p.positionRadius[1]<4.3f && p.positionRadius[1]>=.5625f && std::abs(p.thermalState[3]-10.0f)<.001,"Probe did not advance under gravity");
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
    require(belowLedge>=16,"Live probes did not exercise the lower slope");
    // An occupancy edit invalidates the snapshot before any force/age work.
    store->setMaterial(*terrain::addressOf({0,4,0}),0,store->revision());bool stale=false;
    try{probes.tick(1,world,*store);}catch(const std::invalid_argument&){stale=true;}
    require(stale && probes.ticks()==600,"Stale live probe tick was submitted");const auto unchanged=snapshot();require(std::memcmp(actual.data(),unchanged.data(),sizeof(actual))==0,"Stale probe rejection mutated state");
    const auto& occupiedProbe=actual[0];
    store->setMaterial(*terrain::addressAtRelative({occupiedProbe.positionRadius[0],occupiedProbe.positionRadius[1],occupiedProbe.positionRadius[2]}),0,store->revision());
    GpuVoxelWorld overlapWorld(*store,{});probes.tick(1,overlapWorld,*store);probes.request(6);
    const auto rolledBack=snapshot();require(probes.fault(),"Overlapping terrain did not latch a probe fault");
    require(std::memcmp(actual.data(),rolledBack.data(),sizeof(actual))==0,"Probe failure did not roll back the entire tick");
    std::cout<<"GPU live voxel probes passed: 256 particles, 600 ticks / 2400 substeps, gravity/drag, full basin containment, identity, stale-world rejection and whole-tick rollback\n";
}
}
