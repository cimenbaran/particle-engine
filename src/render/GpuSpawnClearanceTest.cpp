#include "GpuSpawnClearance.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
namespace pw {
namespace {
template<class T>struct ClearanceHandle {T value;explicit ClearanceHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Clearance test allocation failed");}~ClearanceHandle(){bgfx::destroy(value);}ClearanceHandle(const ClearanceHandle&)=delete;};
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
}
void runGpuSpawnClearanceTest(const std::filesystem::path& assets) {
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,2> materials{};materials[1].identity[0]=1;materials[1].identity[1]=1;
    ClearanceHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    ClearanceHandle exporter(loadCompute(assets,"cs_export"));ClearanceHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned cases=0,queries=0;
    auto bodies=[](unsigned count){std::vector<ParticleGpu> p(count);for(unsigned i=0;i<count;++i)p[i]={{0,1,0,.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,i,1}};return p;};
    auto run=[&](const std::vector<ParticleGpu>& input,const Command& spawn,unsigned forcedCode=UINT32_MAX,GpuSpawnClearance* reused=nullptr) {
        const auto count=std::uint32_t(input.size()),storage=std::max(1u,count),requests=std::max(1u,spawn.count);
        auto uploaded=input;if(!count)uploaded.resize(1);
        ClearanceHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(uploaded.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        auto owned=reused?nullptr:std::make_unique<GpuSpawnClearance>(assets,count);auto& clearance=reused?*reused:*owned;
        clearance.submit(0,state.value,materialBuffer.value,2,spawn);
        ClearanceHandle result(bgfx::createTexture2D(1,std::uint16_t(requests),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ClearanceHandle staging(bgfx::createTexture2D(1,std::uint16_t(requests),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ClearanceHandle old(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ClearanceHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(spawn.count){const float p[]{0,float(spawn.count),0,1};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,clearance.output(),bgfx::Access::Read);bgfx::setImage(1,result.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(spawn.count));}
        if(count){const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,state.value,bgfx::Access::Read);bgfx::setImage(1,old.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));}
        std::vector<std::array<std::uint32_t,4>> actual(requests);std::vector<ParticleGpu> original(storage);std::uint32_t ready=0;
        bgfx::TextureRegion src,dst;
        if(spawn.count){src.handle=result.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=bgfx::read(dst,actual.data());}
        if(count){src.handle=old.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        require(!count || !std::memcmp(input.data(),original.data(),count*sizeof(ParticleGpu)),"Clearance modified physical state");
        auto lattice=spawnLattice(spawn.minimum,spawn.maximum);
        for(unsigned request=0;request<spawn.count;++request) {
            unsigned expected=forcedCode;
            if(expected==UINT32_MAX) {
                expected=0;auto p=spawnLatticePosition(lattice,request,std::uint32_t(spawn.sequence));
                if(!p)expected=4;
                else if((*p)[0]<-16 || (*p)[0]>=16 || (*p)[1]<0 || (*p)[1]>=16 || (*p)[2]<-16 || (*p)[2]>=16)expected=3;
                else for(const auto& particle:input) {
                    if(!(particle.identity[1]&1) || particle.identity[0]!=0)continue;
                    long double squared=0;for(unsigned axis=0;axis<3;++axis){const long double d=static_cast<long double>(particle.positionRadius[axis])-(*p)[axis];squared+=d*d;}
                    if(squared<.015625L)expected=1;
                }
            }
            if(actual[request][0]!=expected)throw std::runtime_error("Spawn clearance case "+std::to_string(cases)+" request "+std::to_string(request)+": "+std::to_string(actual[request][0])+" expected "+std::to_string(expected));
            require(actual[request][1]<=count,"Spawn clearance visited duplicate slots");
        }
        if(count==1024 && spawn.count==1 && forcedCode==UINT32_MAX)require(actual[0][1]==1024,"Dense spawn clearance truncated neighbors");
        ++cases;queries+=spawn.count;
    };
    Command point;point.type=CommandType::SpawnBrush;point.count=1;point.minimum=point.maximum={0,1,0};
    for(unsigned count:{0u,1u,127u,128u,129u,255u,256u,257u,1024u})run(bodies(count),point);
    for(unsigned axis=0;axis<3;++axis)for(float direction:{-1.0f,1.0f})for(float distance:{std::nextafter(.125f,0.0f),.125f,std::nextafter(.125f,1.0f),.25f}) {
        auto p=bodies(1);p[0].positionRadius[axis]+=direction*distance;run(p,point);
    }
    auto p=bodies(3);p[0].identity[1]=0;p[0].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();p[1].identity[0]=1;p[2].velocityInvMass[3]=0;run(p,point);
    p[2].identity[1]=0;run(p,point);
    // Subnormal inward/outward offsets are too small for even long-double
    // subtraction at .125. Their strict classification is known analytically.
    for(float sign:{-1.0f,1.0f}) {
        point.minimum=point.maximum={sign*.125f,1,0};p=bodies(1);
        p[0].positionRadius[0]=sign*std::numeric_limits<float>::denorm_min();run(p,point,1);
        p[0].positionRadius[0]=-sign*std::numeric_limits<float>::denorm_min();run(p,point,0);
    }
    std::mt19937 random(99231);p=bodies(257);
    for(auto& particle:p){particle.positionRadius[0]=float(int(random()%31001)-15500)/1000;particle.positionRadius[1]=float(random()%15501)/1000;particle.positionRadius[2]=float(int(random()%31001)-15500)/1000;}
    Command brush=point;brush.count=4096;brush.minimum={-16,0,-16};brush.maximum={15.875f,15.875f,15.875f};
    GpuSpawnClearance reused(assets,257);
    for(unsigned seed:{0u,1u,77u,UINT32_MAX}){brush.sequence=seed;run(p,brush,UINT32_MAX,&reused);}
    brush.minimum={-1,1,-1};brush.maximum={1,3,1};
    for(auto& particle:p){particle.positionRadius[0]=float(int(random()%2001)-1000)/1000;particle.positionRadius[1]=1+float(random()%2001)/1000;particle.positionRadius[2]=float(int(random()%2001)-1000)/1000;}
    for(unsigned seed:{0u,13u,101u,UINT32_MAX}){brush.sequence=seed;run(p,brush,UINT32_MAX,&reused);}
    // A bad radius far from all queries invalidates the complete snapshot.
    point.minimum=point.maximum={0,1,0};p=bodies(257);p.back().positionRadius[0]=15;p.back().positionRadius[3]=.1f;run(p,point,2,&reused);run(p,brush,2,&reused);
    p.back().positionRadius[3]=std::numeric_limits<float>::quiet_NaN();run(p,point,2,&reused);
    p=bodies(257);p.back().identity[3]=0;run(p,point,2,&reused);
    p=bodies(257);p.back().positionRadius[0]=16;run(p,point,2,&reused);
    p=bodies(257);run(p,point,UINT32_MAX,&reused);
    for(auto position:{std::array<float,3>{16,1,0},{-16,1,0},{0,16,0},{0,-.125f,0},{0,1,16}}){point.minimum=point.maximum=position;run({},point);}
    point.minimum={.01f,1,.01f};point.maximum={.1f,1,.1f};point.count=129;run({},point);
    point.count=0;run({},point);
    std::cout<<"GPU spawn clearance passed: "<<cases<<" cases, "<<queries<<" queries, all-pairs oracle, dense uncapped traversal, subnormals, faults/recovery and immutable state\n";
}
}
