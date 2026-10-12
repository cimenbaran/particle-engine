#include "GpuGranularStep.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "assets/Materials.hpp"
#include "../../tests/reference/SweptBounds.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct GranularHandle {
    T value;explicit GranularHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Granular test allocation failed");}
    ~GranularHandle(){bgfx::destroy(value);}GranularHandle(const GranularHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
}
void runGpuGranularStepTest(const std::filesystem::path& assets) {
    bool rejected=false;GranularStepProfile excessive;excessive.speedLimit=60;excessive.maxSubsteps=64;
    try {GpuGranularStep invalid(assets,1,excessive);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Granular owner accepted excessive dispatch substeps");
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,3> materials{};for(unsigned i=0;i<3;++i){materials[i].identity[0]=i;materials[i].identity[1]=i;}
    materials[0].mechanical[1]=0.5f;materials[0].mechanical[2]=0.5f;
    GranularHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    GranularHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    GranularHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    std::array<std::uint32_t,4> stats{};unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> input,unsigned ticks=1,float gravity=9.81f,bool oracle=true,bool error=false,GpuGranularStep* reused=nullptr) {
        const auto count=std::uint32_t(input.size()),storage=std::max(1u,count);
        auto owned=reused?nullptr:std::make_unique<GpuGranularStep>(assets,count,GranularStepProfile{},4,gravity);auto& solver=reused?*reused:*owned;
        auto uploaded=input;if(uploaded.empty())uploaded.resize(1);
        GranularHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(uploaded.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        auto current=state.value;
        // Each frame contains one complete GPU tick. No state readback between
        // ticks; this also checks reuse of both private output buffers.
        for(unsigned tick=0;tick<ticks;++tick){solver.submit(0,current,materialBuffer.value,3);current=solver.output();bgfx::frame();SDL_PumpEvents();}
        GranularHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        GranularHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        GranularHandle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        GranularHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        GranularHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        GranularHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count)for(auto pair:{std::pair{current,image.value},std::pair{state.value,oldImage.value}}){const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));}
        bgfx::setBuffer(0,solver.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(storage),old(storage);bgfx::TextureRegion src,dst;src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count){src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,old.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if((stats[0]!=0)!=error)throw std::runtime_error("Granular error mismatch fixture "+std::to_string(cases)+" mask "+std::to_string(stats[0]));
        for(std::uint32_t i=0;i<count;++i){
            const auto& source=input[i];require(std::memcmp(&old[i],&source,sizeof(source))==0,"Granular step mutated original tick input");
            if(error || !(source.identity[1]&1) || source.identity[0]!=0){require(std::memcmp(&actual[i],&source,sizeof(source))==0,"Granular failure/exclusion changed state");continue;}
            require(actual[i].identity[0]==source.identity[0] && actual[i].identity[2]==source.identity[2] && actual[i].identity[3]==source.identity[3],"Granular identity changed");
            require(actual[i].positionRadius[3]==source.positionRadius[3] && actual[i].velocityInvMass[3]==source.velocityInvMass[3] && actual[i].thermalState[0]==source.thermalState[0] && actual[i].thermalState[1]==source.thermalState[1] && actual[i].thermalState[2]==source.thermalState[2],"Granular persistent field changed");
            if(source.velocityInvMass[3]==0) require(std::memcmp(actual[i].positionRadius,source.positionRadius,32)==0,"Granular static body moved");
            if(oracle){
                reference::SweepState expected;for(unsigned a=0;a<3;++a){expected.position[a]=source.positionRadius[a];expected.velocity[a]=source.velocityInvMass[a];}
                float age=source.thermalState[3];bool alive=true;
                for(unsigned step=0;step<ticks*solver.plan().substeps && alive;++step){
                    const float h=solver.plan().substepSeconds;age+=h;
                    if(source.thermalState[2]>0 && age>=source.thermalState[2]){alive=false;break;}
                    if(source.velocityInvMass[3]==0)continue;
                    expected.velocity[1]-=double(gravity)*h;
                    double speed=0;for(double v:expected.velocity)speed+=v*v;speed=std::sqrt(speed);if(speed>12)for(double& v:expected.velocity)v*=12/speed;
                    expected=reference::sweepBounds(expected,h,0.5,0.5).state;
                }
                require(bool(actual[i].identity[1]&1)==alive && std::abs(actual[i].thermalState[3]-age)<1e-6,"Granular lifetime/age mismatch");
                for(unsigned a=0;a<3;++a){require(std::abs(actual[i].positionRadius[a]-expected.position[a])<2e-4,"Granular free trajectory mismatch");require(std::abs(actual[i].velocityInvMass[a]-expected.velocity[a])<2e-4,"Granular free velocity mismatch");}
            }
            if(source.velocityInvMass[3]!=0 && (actual[i].identity[1]&1)){
                double speed=0;for(unsigned a=0;a<3;++a){const float p=actual[i].positionRadius[a],v=actual[i].velocityInvMass[a];require(std::isfinite(p)&&std::isfinite(v),"Nonfinite granular output");require(p>=(a==1?0.0625f:-7.9375f)&&p<=(a==1?15.9375f:7.9375f),"Granular output outside inset bounds");speed+=double(v)*v;}require(speed<=144.001,"Granular final speed cap exceeded");
            }
        }
        ++cases;actual.resize(count);return actual;
    };
    auto fixture=[](unsigned n){std::vector<ParticleGpu> p(n);for(unsigned i=0;i<n;++i)p[i]={{float(i%8)*0.5f-2,4+float((i/8)%8)*0.5f,float(i/64)*0.5f-1,0.0625f},{0.125f,0,0,1},{300,0.5f,0,0},{0,1,i,1}};return p;};
    for(unsigned n:{0u,1u,127u,128u,129u,255u,256u,257u})check(fixture(n));
    check(fixture(1),10);
    auto p=fixture(1);p[0].positionRadius[1]=0.1f;p[0].velocityInvMass[0]=0;p[0].velocityInvMass[1]=-12;check(p,1,0);require(stats[3]>0,"Granular boundary events missing");
    p=fixture(1);p[0].positionRadius[1]=0.0625f;p[0].velocityInvMass[0]=0;auto rest=check(p,120);require(rest[0].positionRadius[1]==0.0625f && rest[0].velocityInvMass[1]==0,"Resting body drifted");
    p=fixture(1);p[0].velocityInvMass[0]=1000;check(p);require(stats[1]>0,"Granular speed caps missing");
    p=fixture(129);for(unsigned i=0;i<p.size();++i){if(i%4==0)p[i].velocityInvMass[3]=0;if(i%4==1)p[i].identity[0]=1;if(i%4==2)p[i].identity[0]=2;if(i%4==3)p[i].identity[1]=0;}check(p);
    p=fixture(1);p[0].thermalState[2]=0.01f;check(p,3);
    p=fixture(2);p[0].positionRadius[0]=-0.0625f;p[1].positionRadius[0]=0.0625f;p[0].velocityInvMass[0]=1;p[1].velocityInvMass[0]=-1;
    auto pair=check(p,1,0,false);require(stats[2]>0,"Granular contacts missing");require(std::abs(pair[0].velocityInvMass[0]+pair[1].velocityInvMass[0])<1e-5,"Closed pair momentum changed");
    double pairEnergy=0;for(const auto& body:pair)for(unsigned a=0;a<3;++a)pairEnergy+=double(body.velocityInvMass[a])*body.velocityInvMass[a];
    require(pairEnergy<=2.00001,"Closed pair gained kinetic energy");
    require(pair[0].velocityInvMass[0]<0 && pair[1].velocityInvMass[0]>0,"Closed pair did not rebound");
    p=fixture(4);for(unsigned i=0;i<4;++i){p[i].positionRadius[0]=0;p[i].positionRadius[1]=0.0625f+i*0.125f;p[i].velocityInvMass[0]=0;}auto stack=check(p,60,9.81f,false);
    double maxPenetration=0,maxSpeed=0;
    for(unsigned i=0;i<stack.size();++i){
        double speed=0;for(unsigned a=0;a<3;++a)speed+=double(stack[i].velocityInvMass[a])*stack[i].velocityInvMass[a];maxSpeed=std::max(maxSpeed,std::sqrt(speed));
        for(unsigned j=0;j<i;++j){double squared=0;for(unsigned a=0;a<3;++a){const double d=double(stack[i].positionRadius[a])-stack[j].positionRadius[a];squared+=d*d;}maxPenetration=std::max(maxPenetration,0.125-std::sqrt(squared));}
    }
    std::cout<<"Four-body stack after 60 ticks: max penetration "<<maxPenetration<<" m, max speed "<<maxSpeed<<" m/s\n";
    require(maxPenetration<0.00625 && maxSpeed<0.1,"Four-body stack did not settle within the small-fixture limits");
    p=fixture(129);p[7].velocityInvMass[3]=-1;check(p,1,9.81f,false,true);
    p=fixture(129);p[7].velocityInvMass[3]=0;p[7].positionRadius[0]=9;check(p,1,9.81f,false,true);require((stats[0]&4)!=0,"Late contact failure not retained");
    p=fixture(129);p[7].thermalState[0]=std::numeric_limits<float>::quiet_NaN();check(p,1,9.81f,false,true);
    p=fixture(1);p[0].velocityInvMass[0]=std::numeric_limits<float>::max();check(p);
    GpuGranularStep recovery(assets,3);p=fixture(3);p[0].identity[3]=0;check(p,1,9.81f,false,true,&recovery);check(fixture(3),1,9.81f,true,false,&recovery);
    std::cout<<"GPU granular step passed: "<<cases<<" fixtures, full substep chain, analytic motion, resting/stack runs and tick rollback\n";
}
}
