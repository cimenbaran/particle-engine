#include "GpuSweptBounds.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "assets/Materials.hpp"
#include "../../tests/reference/SweptBounds.hpp"
#include "../../tests/reference/SweptSphere.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct SweepHandle {
    T value;explicit SweepHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Sweep test resource unavailable");}
    ~SweepHandle(){bgfx::destroy(value);}SweepHandle(const SweepHandle&)=delete;
};
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
}
void runGpuSweptBoundsTest(const std::filesystem::path& assets) {
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,3> materials{};for(unsigned i=0;i<3;++i){materials[i].identity[0]=i;materials[i].identity[1]=i;}
    SweepHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    SweepHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> input,float dt,float e,float mu,float limit=1000,std::uint32_t budget=128,
                  std::uint32_t error=0,std::uint32_t upstream=0,GpuSweptBounds* reused=nullptr,
                  AnalyticSphere obstacle={},const reference::SweepResult* known=nullptr) {
        const auto count=std::uint32_t(input.size()),storage=std::max(1u,count);
        materials[0].mechanical[1]=mu;materials[0].mechanical[2]=e;
        SweepHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
        auto upload=input;if(upload.empty())upload.resize(1);
        SweepHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(upload.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        std::array<std::uint32_t,4> initial{upstream,0,0,0};
        SweepHandle status(bgfx::createDynamicIndexBuffer(bgfx::copy(initial.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        auto owned=reused?nullptr:std::make_unique<GpuSweptBounds>(assets,count);auto& sweep=reused?*reused:*owned;
        sweep.submit(0,state.value,materialBuffer.value,3,status.value,dt,limit,0.5f,budget,obstacle);
        SweepHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        SweepHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        SweepHandle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        SweepHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        SweepHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        SweepHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count)for(auto pair:{std::pair{sweep.output(),image.value},std::pair{state.value,oldImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);
            bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,sweep.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(storage),old(storage);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count){src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));
            src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,old.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if(stats[0]!=error)throw std::runtime_error("Sweep error mismatch fixture "+std::to_string(cases)+": "+std::to_string(stats[0]));
        std::uint32_t capped=0,events=0,planes=0;
        for(std::uint32_t i=0;i<count;++i) {
            const auto& p=input[i];require(std::memcmp(&old[i],&p,sizeof(p))==0,"Sweep mutated input");
            if(error || !(p.identity[1]&1) || p.identity[0]!=0 || p.velocityInvMass[3]==0) {
                require(std::memcmp(&actual[i],&p,sizeof(p))==0,"Sweep failure/exclusion changed state");continue;
            }
            reference::SweepState start;double squared=0;
            for(unsigned a=0;a<3;++a){start.position[a]=p.positionRadius[a];start.velocity[a]=p.velocityInvMass[a];squared+=start.velocity[a]*start.velocity[a];}
            const double speed=std::sqrt(squared);
            if(speed>limit){for(double& v:start.velocity)v*=double(limit)/speed;++capped;}
            reference::SweepResult expected;
            if(known)expected=*known;
            else if(obstacle.radius>0){
                const reference::SphereCollider collider{{obstacle.centre[0],obstacle.centre[1],obstacle.centre[2]},obstacle.radius};
                const auto result=reference::sweepSphere(start,dt,p.positionRadius[3],collider,e,mu,0.5);
                expected={result.state,result.hit?1u:0u,result.hit?1u:0u};
            }else expected=reference::sweepBounds(start,dt,e,mu,0.5,{},budget);
            events+=expected.events;planes+=expected.planes;
            double finalSpeed=0,initialEnergy=0;
            for(unsigned a=0;a<3;++a) {
                auto mismatch=[&](const char* field) {
                    std::ostringstream message;message<<std::setprecision(17)
                        <<"GPU sweep "<<field<<" differs from CPU reference fixture "<<cases<<" slot "<<i<<" axis "<<a
                        <<" dt "<<dt<<" restitution "<<e<<" friction "<<mu
                        <<" GPU events "<<stats[2]<<" incident normals "<<stats[3];
                    for(unsigned axis=0;axis<3;++axis)message<<" | axis "<<axis
                        <<" initial p/v "<<p.positionRadius[axis]<<"/"<<p.velocityInvMass[axis]
                        <<" actual p/v "<<actual[i].positionRadius[axis]<<"/"<<actual[i].velocityInvMass[axis]
                        <<" expected p/v "<<expected.state.position[axis]<<"/"<<expected.state.velocity[axis];
                    throw std::runtime_error(message.str());
                };
                if(!std::isfinite(actual[i].positionRadius[a]) || !(std::abs(actual[i].positionRadius[a]-expected.state.position[a])<2e-4))mismatch("position");
                if(!std::isfinite(actual[i].velocityInvMass[a]) || !(std::abs(actual[i].velocityInvMass[a]-expected.state.velocity[a])<2e-4*std::max(1.0,std::abs(expected.state.velocity[a]))))mismatch("velocity");
                const double v=actual[i].velocityInvMass[a];finalSpeed+=v*v;initialEnergy+=start.velocity[a]*start.velocity[a];
                const double lo=a==1?0.0625:-7.9375,hi=a==1?15.9375:7.9375;
                require(actual[i].positionRadius[a]>=lo && actual[i].positionRadius[a]<=hi,"GPU sweep escaped bounds");
            }
            if(obstacle.radius>0){double distance=0;for(unsigned axis=0;axis<3;++axis){const double d=actual[i].positionRadius[axis]-obstacle.centre[axis];distance+=d*d;}
                require(std::sqrt(distance)>=obstacle.radius+0.0625-1e-6,"Sphere sweep ended inside obstacle");}
            require(finalSpeed<=initialEnergy+1e-4*std::max(1.0,initialEnergy) && finalSpeed<=double(limit)*limit*1.00001,"Sweep grew energy/exceeded cap");
            require(std::memcmp(&actual[i].positionRadius[3],&p.positionRadius[3],4)==0 &&
                std::memcmp(&actual[i].velocityInvMass[3],&p.velocityInvMass[3],sizeof(p)-28)==0,"Sweep changed persistent fields");
        }
        if(!error)require(stats[1]==capped && stats[2]==events && stats[3]==planes,"Sweep diagnostics differ from reference");
        ++cases;
    };
    auto fixture=[](unsigned n){std::vector<ParticleGpu> p(n);for(unsigned i=0;i<n;++i){p[i]={{0,8,0,0.0625f},{1,-2,3,1},{300,0.5f,0,7},{0,1,i,2}};}return p;};
    for(unsigned n:{0u,1u,127u,128u,129u,255u,256u,257u})check(fixture(n),1.0f/60,0.5f,0.5f);
    std::mt19937 random(20261013);std::uniform_real_distribution<float> pos(-7.5f,7.5f),vel(-500,500);
    auto randoms=fixture(257);for(auto& p:randoms)for(unsigned a=0;a<3;++a){p.positionRadius[a]=pos(random)+(a==1?8:0);p.velocityInvMass[a]=vel(random);}
    for(float e:{0.0f,0.5f,1.0f})for(float mu:{0.0f,0.5f,1.0f})check(randoms,0.2f,e,mu);
    auto p=fixture(1);p[0].positionRadius[1]=1.0625f;p[0].velocityInvMass[0]=p[0].velocityInvMass[2]=0;p[0].velocityInvMass[1]=-10;
    check(p,0.2f,0.5f,0);check(p,0.1f,0.5f,0);
    p[0].positionRadius[1]=0.0625f;p[0].velocityInvMass[1]=-0.1f;check(p,0.1f,1,0);
    p[0].velocityInvMass[0]=1;p[0].velocityInvMass[1]=0;check(p,0.1f,1,1);
    p[0].velocityInvMass[1]=-1;check(p,0.1f,0,1);check(p,0.1f,0,0.25f);
    p=fixture(1);p[0].positionRadius[0]=p[0].positionRadius[2]=7.9375f;p[0].velocityInvMass[0]=p[0].velocityInvMass[2]=2;check(p,0.1f,1,0);check(p,0.1f,0.5f,0.5f);
    check(randoms,1.0f/240,0.5f,0.5f,12);
    p=fixture(1);p[0].velocityInvMass[0]=std::numeric_limits<float>::max();check(p,1.0f/240,0.5f,0.5f,12);
    p=fixture(129);for(unsigned i=0;i<p.size();++i){if(i%3==0)p[i].velocityInvMass[3]=0;else p[i].identity[0]=i%3;}check(p,0.1f,0.5f,0.5f);
    p=fixture(129);p[7].identity[1]=0;p[7].identity[0]=UINT32_MAX;p[7].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(p,0.1f,0.5f,0.5f);
    p=fixture(129);p[7].positionRadius[0]=9;check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].velocityInvMass[0]=std::numeric_limits<float>::infinity();check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].velocityInvMass[0]=1000;check(p,1,1,0,1000,1,4);
    check(fixture(129),0.1f,0,0,1000,128,1,7);check(fixture(0),0.1f,0,0,1000,128,1,7);
    p=fixture(129);p[7].positionRadius[3]=0.125f;check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].velocityInvMass[3]=-1;check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].identity[2]=8;check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].identity[0]=99;check(p,0.1f,0,0,1000,128,2);
    p=fixture(129);p[7].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(p,0.1f,0,0,1000,128,2);
    check(fixture(1),0.1f,1.01f,0,1000,128,2);
    check(fixture(1),0.1f,0,-std::numeric_limits<float>::denorm_min(),1000,128,2);
    check(fixture(1),0.1f,0,std::numeric_limits<float>::quiet_NaN(),1000,128,2);
    p=fixture(1);p[0].velocityInvMass[3]=-0.0f;p[0].positionRadius[0]=9;check(p,0.1f,0,0);
    GpuSweptBounds reused(assets,129);p=fixture(129);p[7].identity[3]=0;check(p,0.1f,0,0,1000,128,2,0,&reused);check(fixture(129),0.1f,0,0,1000,128,0,0,&reused);
    const AnalyticSphere obstacle{{0,8,0},0.9375f};
    auto sphereFixture=[&](unsigned count){auto values=fixture(count);for(auto& body:values){body.positionRadius[0]=-2;body.velocityInvMass[0]=10;body.velocityInvMass[1]=body.velocityInvMass[2]=0;}return values;};
    for(unsigned count:{0u,1u,127u,128u,129u,255u,256u,257u})check(sphereFixture(count),0.2f,0.5f,0.5f,1000,128,0,0,nullptr,obstacle);
    p=sphereFixture(257);for(unsigned i=0;i<p.size();++i)p[i].positionRadius[1]=8+(float(i%17)-8)*0.1f;
    for(float restitution:{0.0f,0.5f,1.0f})for(float friction:{0.0f,0.5f,1.0f})check(p,0.2f,restitution,friction,1000,128,0,0,nullptr,obstacle);
    for(unsigned axis=0;axis<3;++axis)for(float sign:{-1.0f,1.0f}){
        p=sphereFixture(1);for(unsigned a=0;a<3;++a){p[0].positionRadius[a]=obstacle.centre[a];p[0].velocityInvMass[a]=0;}
        p[0].positionRadius[axis]+=2*sign;p[0].velocityInvMass[axis]=-10*sign;
        check(p,0.2f,0.5f,0.25f,1000,128,0,0,nullptr,obstacle);
        check(p,0.1f,0.5f,0.25f,1000,128,0,0,nullptr,obstacle);
    }
    // Oblique friction is sensitive to a non-unit contact normal. Rotate the
    // failing half-radius miss across axes and octants; keep the original oracle tolerances.
    for(unsigned axis=0;axis<3;++axis)for(float sign:{-1.0f,1.0f})for(float friction:{0.49f,0.5f,0.57f}) {
        p=sphereFixture(1);for(unsigned a=0;a<3;++a){p[0].positionRadius[a]=obstacle.centre[a];p[0].velocityInvMass[a]=0;}
        p[0].positionRadius[axis]-=2*sign;p[0].positionRadius[(axis+1)%3]+=0.5f;p[0].velocityInvMass[axis]=10*sign;
        check(p,0.2f,0,friction,1000,128,0,0,nullptr,obstacle);
    }
    for(unsigned signs=0;signs<8;++signs)for(float friction:{0.49f,0.5f,0.57f}) {
        p=sphereFixture(1);const float sx=(signs&1)?1.0f:-1.0f,sy=(signs&2)?1.0f:-1.0f,sz=(signs&4)?1.0f:-1.0f;
        const std::array<float,3> direction{sx/std::sqrt(3.0f),sy/std::sqrt(3.0f),sz/std::sqrt(3.0f)};
        const std::array<float,3> miss{sx/std::sqrt(2.0f),-sy/std::sqrt(2.0f),0};
        for(unsigned a=0;a<3;++a){p[0].positionRadius[a]=obstacle.centre[a]-2*direction[a]+0.5f*miss[a];p[0].velocityInvMass[a]=10*direction[a];}
        check(p,0.2f,0,friction,1000,128,0,0,nullptr,obstacle);
    }
    p=sphereFixture(1);check(p,0.1f,0.5f,0,1000,128,0,0,nullptr,obstacle);
    p[0].positionRadius[1]=9;check(p,0.4f,1,1,1000,128,0,0,nullptr,obstacle); // grazing
    p[0].positionRadius[1]=9.01f;check(p,0.4f,1,1,1000,128,0,0,nullptr,obstacle); // miss
    p=sphereFixture(1);p[0].positionRadius[0]=-1;p[0].velocityInvMass[0]=0.1f;check(p,0.1f,1,0,1000,128,0,0,nullptr,obstacle);
    p=sphereFixture(1);p[0].positionRadius[0]=-4;p[0].velocityInvMass[0]=10000;check(p,0.01f,0,0,10000,128,0,0,nullptr,obstacle);
    // Independent 1D path lengths: sphere then wall, and wall then sphere.
    p=sphereFixture(1);p[0].positionRadius[0]=-4;
    reference::SweepResult twoHits{{{-7.875,8,0},{10,0,0}},2,2};
    check(p,1,1,0,1000,128,0,0,nullptr,obstacle,&twoHits);
    check(p,1,1,0,1000,1,4,0,nullptr,obstacle);
    p[0].positionRadius[0]=-7;p[0].velocityInvMass[0]=-10;
    twoHits={{{-3.125,8,0},{-10,0,0}},2,2};check(p,1,1,0,1000,128,0,0,nullptr,obstacle,&twoHits);
    p=sphereFixture(129);p[7].positionRadius[0]=0;check(p,0.2f,0,0,1000,128,2,0,&reused,obstacle);
    check(sphereFixture(129),0.2f,0,0,1000,128,0,0,&reused,obstacle);
    check(fixture(129),0.1f,0,0,1000,128,0,0,&reused); // disabled obstacle after reuse
    p=sphereFixture(129);for(unsigned i=0;i<p.size();++i){if(i%4==0)p[i].velocityInvMass[3]=0;if(i%4==1)p[i].identity[0]=1;if(i%4==2)p[i].identity[0]=2;if(i%4==3)p[i].identity[1]=0;}
    check(p,0.2f,0,0,1000,128,0,0,nullptr,obstacle);
    check(sphereFixture(129),0.2f,0,0,1000,128,1,1,nullptr,obstacle);
    std::cout<<"GPU swept bounds passed: "<<cases<<" fixtures, CPU trajectories, speed caps, corner response, containment and whole-pass rollback\n";
}
}
