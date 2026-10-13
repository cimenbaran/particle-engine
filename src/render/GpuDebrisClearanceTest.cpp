#include "GpuDebrisClearance.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
namespace pw {
namespace {
template<class T>struct DebrisCheckHandle {
    T value;explicit DebrisCheckHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Debris clearance test allocation failed");}
    ~DebrisCheckHandle(){bgfx::destroy(value);}DebrisCheckHandle(const DebrisCheckHandle&)=delete;
};
}
void runGpuDebrisClearanceTest(const std::filesystem::path& assets) {
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    const auto materials=loadMaterials(assets/"materials/default.json");const auto packed=materials.packed();
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    DebrisCheckHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),std::uint32_t(packed.size()*64)),layout,BGFX_BUFFER_COMPUTE_READ));
    DebrisCheckHandle exporter(loadCompute(assets,"cs_export"));DebrisCheckHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    terrain::SparseVoxelStore store(terrain::ResidentChunkLayout({-1,0,-1},{2,1,2}),std::uint32_t(packed.size()));
    GpuVoxelWorld world(store,{});VoxelContactContext context{world,store,{}};
    unsigned cases=0,queries=0;
    auto read=[&](bgfx::DynamicVertexBufferHandle buffer,unsigned count,unsigned lanes) {
        DebrisCheckHandle image(bgfx::createTexture2D(std::uint16_t(lanes),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        DebrisCheckHandle staging(bgfx::createTexture2D(std::uint16_t(lanes),std::uint16_t(std::max(1u,count)),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        const float p[]{0,float(count),0,float(lanes)};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,buffer,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        if(count)bgfx::dispatch(5,exporter.value,dispatchGroups(count));
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(7,dst,src);
        std::vector<std::uint32_t> result(std::max(1u,count)*lanes*4);const auto ready=bgfx::read(dst,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();return result;
    };
    auto seed=[](float x=0,float y=2,float z=0){DebrisSeed s;s.position={x,y,z};return s;};
    auto bodies=[](unsigned count){std::vector<ParticleGpu> p(count);for(unsigned i=0;i<count;++i)p[i]={{0,2,0,.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,i,1}};return p;};
    auto run=[&](std::vector<ParticleGpu> input,const std::vector<DebrisSeed>& seeds,const VoxelContactContext& terrain,unsigned forced=UINT32_MAX,unsigned visited=UINT32_MAX,unsigned claimFault=0) {
        const unsigned capacity=unsigned(input.size()),count=unsigned(seeds.size());auto uploaded=input;if(uploaded.empty())uploaded.resize(1);
        DebrisCheckHandle source(bgfx::createDynamicVertexBuffer(bgfx::copy(uploaded.data(),std::uint32_t(uploaded.size()*64)),layout,BGFX_BUFFER_COMPUTE_READ_WRITE));
        GpuDebrisReservation reservation(assets,capacity);GpuDebrisActivation activation(assets,capacity,materials,seeds);GpuDebrisClearance clearance(assets,capacity);
        bool rejected=false;try{clearance.submit(2,activation,materialBuffer.value,unsigned(packed.size()),terrain);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Unsubmitted debris activation accepted");
        reservation.submit(0,source.value,count);activation.submit(1,reservation);
        const auto before=read(activation.output(),capacity,4);
        if(claimFault) {
            std::array<std::uint32_t,4> claim{0,2,1,0};
            if(claimFault==1)claim[0]=capacity;
            if(claimFault==2)claim[1]=3;
            if(claimFault==3)claim[3]=1;
            bgfx::update(reservation.claims(),0,bgfx::copy(claim.data(),16));
        }
        clearance.submit(2,activation,materialBuffer.value,unsigned(packed.size()),terrain);
        PoolDiagnostics diagnostics(assets);require(diagnostics.request(6,clearance.status(),cases),"Clearance diagnostics busy");
        const auto actual=read(clearance.records(),count,3);diagnostics.poll(bgfx::frame());
        const auto after=read(activation.output(),capacity,4),original=read(source.value,capacity,4);
        require(!capacity || before==after,"Clearance mutated activated state");require(!capacity || std::memcmp(original.data(),input.data(),capacity*64)==0,"Clearance mutated live state");
        unsigned errors=0,clear=0;std::vector<std::array<float,3>> solids;
        for(const auto& p:input)if(p.identity[1]==1 && packed[p.identity[0]].identity[1]==0)solids.push_back({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]});
        for(unsigned i=0;i<count;++i) {
            unsigned expected=forced;
            if(expected==UINT32_MAX) {
                expected=0;
                auto overlap=[&](const std::array<float,3>& p){long double d2=0;for(unsigned a=0;a<3;++a){const long double d=static_cast<long double>(p[a])-seeds[i].position[a];d2+=d*d;}return d2<.015625L;};
                for(const auto& p:solids)if(overlap(p))expected|=8;
                for(unsigned j=0;j<count;++j)if(j!=i && overlap(seeds[j].position))expected|=8;
            }
            const auto got=actual[i*12+8];if(got!=expected)throw std::runtime_error("Debris clearance case "+std::to_string(cases)+" query "+std::to_string(i)+": "+std::to_string(got)+" expected "+std::to_string(expected));
            if(visited!=UINT32_MAX)require(actual[i*12+9]==visited,"Debris traversal truncated or duplicated neighbors");
            errors|=expected;if(!expected)++clear;
        }
        const auto& status=diagnostics.latest();require(status.valid && status.active==errors && status.free==count && status.reserved==clear && status.rejected==count-clear,"Debris clearance aggregate mismatch");
        ++cases;queries+=count;
    };
    run({}, {},context);
    for(unsigned capacity:{1u,127u,128u,129u,255u,256u,257u,4096u}) {
        auto p=bodies(capacity);p[0].identity[1]=0;run(p,{seed()},context,capacity==1?0:8,capacity-1);
        for(auto& particle:p)particle.identity[1]=0;
        std::vector<DebrisSeed> seeds;for(unsigned i=0;i<std::min(capacity,512u);++i)seeds.push_back(seed(-2+float(i%32)*.125f,1+float(i/32)*.125f));
        run(p,seeds,context);
    }
    {
        auto p=bodies(4096);for(auto& particle:p)particle.identity[1]=0;
        std::vector<DebrisSeed> seeds;for(unsigned i=0;i<4096;++i)seeds.push_back(seed(-2+float(i%32)*.125f,1+float((i/32)%16)*.125f,-1+float(i/512)*.125f));
        run(p,seeds,context);
    }
    for(float sign:{-1.f,1.f})for(float distance:{std::nextafter(.125f,0.f),.125f,std::nextafter(.125f,1.f)}) {
        auto p=bodies(2);p[0].identity[1]=0;p[1].positionRadius[0]=sign*distance;run(p,{seed()},context);
        p[1].identity[1]=0;run(p,{seed(),seed(sign*distance)},context);
    }
    for(float sign:{-1.f,1.f})for(float inward:{-1.f,1.f}) {
        const auto tiny=sign*inward*std::numeric_limits<float>::denorm_min();auto p=bodies(2);p[0].identity[1]=0;p[1].positionRadius[0]=sign*.125f;
        run(p,{seed(tiny)},context,inward>0?8:0);
        p[1].positionRadius[0]=tiny;run(p,{seed(sign*.125f)},context,inward>0?8:0);
        p[1].identity[1]=0;run(p,{seed(tiny),seed(sign*.125f)},context,inward>0?8:0);
    }
    auto p=bodies(4);p[0].identity[1]=0;p[1].identity[1]=0;p[1].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();p[2].identity[0]=2;p[3].velocityInvMass[3]=0;
    run(p,{seed()},context,8,1);p[3].identity[1]=0;run(p,{seed()},context,0,0);
    p=bodies(2);p[0].identity[1]=0;p[1].positionRadius[0]=3;p[1].positionRadius[3]=.1f;run(p,{seed()},context,4);
    for(unsigned fault=1;fault<=3;++fault){auto invalid=bodies(1);invalid[0].identity[1]=0;run(invalid,{seed()},context,2,UINT32_MAX,fault);}
    run(bodies(1),{seed()},context,1);run({}, {seed()},context,1);
    std::mt19937 random(99101);p=bodies(257);
    for(unsigned trial=0;trial<4;++trial) {
        for(unsigned i=0;i<p.size();++i){p[i].identity[1]=i<129?0:1;p[i].positionRadius[0]=float(int(random()%4001)-2000)/1000;p[i].positionRadius[1]=1+float(random()%2001)/1000;p[i].positionRadius[2]=float(int(random()%4001)-2000)/1000;}
        std::vector<DebrisSeed> seeds;for(unsigned i=0;i<129;++i)seeds.push_back(seed(float(int(random()%4001)-2000)/1000,1+float(random()%2001)/1000,float(int(random()%4001)-2000)/1000));run(p,seeds,context);
    }
    terrain::SparseVoxelStore occupied(terrain::ResidentChunkLayout({-1,0,-1},{2,1,2}),unsigned(packed.size()));
    const terrain::VoxelAddress voxel{{0,0,0},{8,8,8}};occupied.setMaterial(voxel,0,0);GpuVoxelWorld solidWorld(occupied,{});VoxelContactContext solid{solidWorld,occupied,{}};
    p=bodies(1);p[0].identity[1]=0;
    run(p,{seed(1.0625f,1.1875f,1.0625f)},solid,0);run(p,{seed(1.0625f,std::nextafter(1.1875f,0.f),1.0625f)},solid,16);
    run(p,{seed(1.0625f,1.0625f,1.0625f)},solid,16);run(p,{seed(0,4,0)},context,16);
    occupied.setMaterial(voxel,terrain::SparseVoxelStore::empty,occupied.revision());GpuVoxelWorld removedWorld(occupied,{});VoxelContactContext removed{removedWorld,occupied,{}};
    run(p,{seed(1.0625f,1.0625f,1.0625f)},removed,0);
    // Host rejection must precede any dispatch, even for an empty transaction.
    DebrisCheckHandle source(bgfx::createDynamicVertexBuffer(4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE));
    GpuDebrisReservation reservation(assets,0);reservation.submit(0,source.value,0);GpuDebrisActivation activation(assets,0,materials,{});activation.submit(1,reservation);GpuDebrisClearance clearance(assets,0);
    for(const auto& invalid:{solid,VoxelContactContext{world,occupied,{}},VoxelContactContext{world,store,{1,0,0}}}) {
        bool rejected=false;try{clearance.submit(2,activation,materialBuffer.value,unsigned(packed.size()),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Stale, foreign or displaced terrain accepted");
    }
    for(auto invalid:{bgfx::DynamicVertexBufferHandle BGFX_INVALID_HANDLE,activation.output(),activation.claims(),world.buffer()}) {
        bool rejected=false;try{clearance.submit(2,activation,invalid,unsigned(packed.size()),context);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Invalid or aliased debris material buffer accepted");
    }
    for(unsigned count:{0u,65537u}) {
        bool rejected=false;try{clearance.submit(2,activation,materialBuffer.value,count,context);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Invalid debris material count accepted");
    }
    GpuDebrisClearance wrongCapacity(assets,1);bool rejected=false;
    try{wrongCapacity.submit(2,activation,materialBuffer.value,unsigned(packed.size()),context);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Mismatched debris clearance capacity accepted");
    read(activation.output(),0,4);
    std::cout<<"GPU debris clearance passed: "<<cases<<" batches, "<<queries<<" queries, all-pairs oracle, uncapped dense traversal, strict tangency/subnormals, terrain revisions and immutable state\n";
}
}
