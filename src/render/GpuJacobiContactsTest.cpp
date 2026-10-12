#include "GpuJacobiContacts.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/SpatialGrid.hpp"
#include "core/ComputeDispatch.hpp"
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
template<class T> struct ContactHandle {
    T value;
    explicit ContactHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Contact test allocation failed");}
    ~ContactHandle(){bgfx::destroy(value);}
    ContactHandle(const ContactHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
struct ReferenceStep {std::vector<ParticleGpu> state;std::uint32_t error=0,maxContacts=0;};
ReferenceStep reference(const std::vector<ParticleGpu>& input,float relaxation,float limit,bool bounded=false) {
    ReferenceStep result{input};const auto count=std::uint32_t(input.size());
    const SpatialGrid grid({0,0,0},0.125f,{-64,0,-64},{128,128,128});
    std::vector<bool> included(count);
    for(std::uint32_t i=0;i<count;++i) {
        const auto& p=input[i];if(!(p.identity[1]&1)) continue;
        if(p.identity[0]>=4) {result.error|=1;continue;}
        if(p.identity[0]==1 || p.identity[0]==2) continue;
        included[i]=true;
        if(p.identity[2]!=i || !p.identity[3] || !grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]})) result.error|=1;
    }
    if(result.error) return result;
    for(std::uint32_t i=0;i<count;++i) if(included[i]) {
        const auto& p=input[i];const auto mass=p.velocityInvMass[3];
        if(p.positionRadius[3]!=GpuJacobiContacts::radius || !std::isfinite(mass) || mass<0 || (mass!=0 && mass<std::numeric_limits<float>::min())) result.error|=4;
    }
    if(result.error) return result;
    if(bounded) for(std::uint32_t i=0;i<count;++i) if(included[i] && input[i].velocityInvMass[3]!=0)
        for(unsigned axis=0;axis<3;++axis) if(input[i].positionRadius[axis]<(axis==1?0.0625f:-7.9375f) || input[i].positionRadius[axis]>(axis==1?15.9375f:7.9375f)) result.error|=16;
    if(result.error) return result;
    for(std::uint32_t i=0;i<count;++i) if(included[i]) {
        const auto& a=input[i];std::array<double,3> correction{};std::uint32_t contacts=0;
        for(std::uint32_t j=0;j<count;++j) if(j!=i && included[j]) {
            const auto& b=input[j];std::array<double,3> delta{};double squared=0;
            for(unsigned axis=0;axis<3;++axis) {delta[axis]=double(b.positionRadius[axis])-a.positionRadius[axis];squared+=delta[axis]*delta[axis];}
            const double distance=std::sqrt(squared),penetration=2*GpuJacobiContacts::radius-distance;
            if(penetration<=0) continue;
            ++contacts;const double wi=a.velocityInvMass[3],wj=b.velocityInvMass[3];if(wi==0 || wi+wj==0) continue;
            std::array<double,3> direction{};
            if(squared>1e-12) for(unsigned axis=0;axis<3;++axis) direction[axis]=-delta[axis]/distance;
            else direction[(std::min(i,j)*1664525u+std::max(i,j)*1013904223u)%3]=i<j?-1:1;
            for(unsigned axis=0;axis<3;++axis) correction[axis]+=direction[axis]*penetration*wi/(wi+wj);
        }
        result.maxContacts=std::max(result.maxContacts,contacts);
        double squared=0;for(auto& x:correction) {x*=relaxation;squared+=x*x;}
        if(squared>double(limit)*limit) for(auto& x:correction) x*=limit/std::sqrt(squared);
        for(unsigned axis=0;axis<3;++axis) result.state[i].positionRadius[axis]=float(double(a.positionRadius[axis])+correction[axis]);
        if(bounded && a.velocityInvMass[3]!=0) for(unsigned axis=0;axis<3;++axis)
            result.state[i].positionRadius[axis]=std::clamp(result.state[i].positionRadius[axis],axis==1?0.0625f:-7.9375f,axis==1?15.9375f:7.9375f);
        const auto& p=result.state[i];if(!grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]})) result.error|=16;
    }
    if(result.error) result.state=input;
    return result;
}
}
void runGpuJacobiContactsTest(const std::filesystem::path& assets) {
    std::array<MaterialGpu,4> materials{};for(unsigned i=0;i<4;++i) {materials[i].identity[0]=i;materials[i].identity[1]=i==3?0:i;}
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    ContactHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    ContactHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    ContactHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> input,unsigned iterations=1,float relaxation=0.8f,float limit=GpuJacobiContacts::radius*0.5f,GpuJacobiContacts* reused=nullptr,bool bounded=false) {
        const auto count=std::uint32_t(input.size());
        auto owned=reused?nullptr:std::make_unique<GpuJacobiContacts>(assets,count,relaxation,limit,bounded);
        auto& solver=reused?*reused:*owned;
        ContactHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        ContactHandle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ContactHandle originalImage(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle originalStaging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ContactHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        auto expected=input;ReferenceStep last;auto current=state.value;
        for(unsigned iteration=0;iteration<iterations;++iteration) {
            last=reference(expected,relaxation,limit,bounded);expected=last.state;
            solver.submit(0,current,materialBuffer.value,std::uint32_t(materials.size()));current=solver.output();
        }
        const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);bgfx::setBuffer(0,current,bgfx::Access::Read);
        bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setImage(1,originalImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        bgfx::setBuffer(0,solver.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(count),original(count);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=originalImage.value;dst.handle=originalStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
        src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,stats.data()));
        while(bgfx::frame()<ready) SDL_PumpEvents();
        require(std::memcmp(original.data(),input.data(),count*sizeof(ParticleGpu))==0,"Jacobi mutated its immutable input");
        if(stats[0]!=last.error) throw std::runtime_error("Jacobi error mismatch actual "+std::to_string(stats[0])+" expected "+std::to_string(last.error));
        if(!last.error && iterations==1) require(stats[1]==last.maxContacts,"Jacobi overlap metric mismatch");
        require(stats[1]<count && stats[2]<=count && stats[3]<=27*64,"Jacobi diagnostics outside bounds");
        for(std::uint32_t i=0;i<count;++i) {
            require(std::memcmp(&actual[i].positionRadius[3],&input[i].positionRadius[3],sizeof(ParticleGpu)-3*sizeof(float))==0,"Jacobi changed radius, velocity, thermal state or identity");
            if(last.error || !(input[i].identity[1]&1) || input[i].identity[0]==1 || input[i].identity[0]==2 || input[i].velocityInvMass[3]==0) {
                require(std::memcmp(&actual[i],&input[i],sizeof(ParticleGpu))==0,"Jacobi modified failed/excluded/static state");continue;
            }
            double displacement=0;
            for(unsigned axis=0;axis<3;++axis) {
                require(std::isfinite(actual[i].positionRadius[axis]) && std::abs(actual[i].positionRadius[axis]-expected[i].positionRadius[axis])<2e-5f,"Jacobi CPU-reference position mismatch");
                if(bounded) require(actual[i].positionRadius[axis]>=(axis==1?0.0625f:-7.9375f) &&
                    actual[i].positionRadius[axis]<=(axis==1?15.9375f:7.9375f),"Bounded Jacobi escaped inset planes");
                const double d=double(actual[i].positionRadius[axis])-input[i].positionRadius[axis];displacement+=d*d;
            }
            require(std::sqrt(displacement)<=double(iterations)*limit+2e-6,"Jacobi correction bound exceeded");
        }
        ++cases;
    };
    auto fixture=[](std::uint32_t count,bool dense=false) {
        std::vector<ParticleGpu> p(count);
        for(std::uint32_t i=0;i<count;++i) {
            p[i].positionRadius[0]=dense?0.f:float(i%8)*0.046875f;
            p[i].positionRadius[1]=4.f+(dense?0.f:float((i/8)%8)*0.046875f);
            p[i].positionRadius[2]=dense?0.f:float(i/64)*0.046875f;
            p[i].positionRadius[3]=GpuJacobiContacts::radius;p[i].velocityInvMass[0]=0.25f;p[i].velocityInvMass[3]=1;
            p[i].thermalState[0]=300;p[i].thermalState[1]=0.5f;p[i].identity[0]=i%2?3:0;
            p[i].identity[1]=1;p[i].identity[2]=i;p[i].identity[3]=i+1;
        }return p;
    };
    for(std::uint32_t count:{1u,2u,127u,128u,129u,255u,256u,257u}) check(fixture(count));
    for(std::uint32_t count:{2u,129u,257u}) {check(fixture(count,true));check(fixture(count),8);}
    auto pair=fixture(2);pair[1].positionRadius[0]=0.0625f;
    check(pair,1,1,0.0625f);pair[0].velocityInvMass[3]=0.25f;check(pair,1,1,0.0625f);
    pair[0].velocityInvMass[3]=0;check(pair,1,1,0.0625f);pair[1].velocityInvMass[3]=0;check(pair);
    pair=fixture(2);pair[1].positionRadius[0]=0.125f;check(pair);pair[1].positionRadius[0]=1;check(pair);
    auto mixed=fixture(129);for(unsigned i=0;i<mixed.size();++i) {
        if(i%7==1) mixed[i].identity[0]=1;
        if(i%7==2) mixed[i].identity[0]=2;
        if(i%7==3) {mixed[i].identity[1]=0;mixed[i].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();}
        if(i%7==4) mixed[i].velocityInvMass[3]=0;
    }check(mixed);check(mixed,4);
    auto bad=fixture(3);bad[0].positionRadius[3]=0.05f;check(bad);
    for(float mass:{-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::denorm_min()}) {bad=fixture(3);bad[0].velocityInvMass[3]=mass;check(bad);}
    bad=fixture(3);bad[0].positionRadius[0]=8;check(bad);
    bad=fixture(3);bad[0].identity[3]=0;check(bad);
    bad=fixture(2);bad[0].positionRadius[1]=0;bad[1].positionRadius[1]=0.03125f;check(bad);
    pair=fixture(2);pair[0].velocityInvMass[3]=std::numeric_limits<float>::max();pair[1].velocityInvMass[3]=std::numeric_limits<float>::max();check(pair);
    pair[0].velocityInvMass[3]=std::numeric_limits<float>::min();pair[1].velocityInvMass[3]=std::numeric_limits<float>::min();check(pair);
    pair[1].velocityInvMass[3]=std::numeric_limits<float>::max();check(pair);
    std::swap(pair[0].velocityInvMass[3],pair[1].velocityInvMass[3]);check(pair);
    // Opposed pair corrections into every inset plane, including repeated
    // rebuilds and a corner. Projection must retain the displacement bound.
    for(unsigned axis=0;axis<3;++axis) for(bool high:{false,true}) {
        pair=fixture(2);pair[0].positionRadius[0]=pair[1].positionRadius[0]=0;
        const float wall=axis==1?(high?15.9375f:0.0625f):(high?7.9375f:-7.9375f);
        pair[0].positionRadius[axis]=wall;pair[1].positionRadius[axis]=wall+(high?-0.0625f:0.0625f);
        check(pair,1,0.8f,0.03125f,nullptr,true);check(pair,8,0.8f,0.03125f,nullptr,true);
    }
    pair=fixture(2);pair[0].positionRadius[0]=pair[0].positionRadius[2]=-7.9375f;
    pair[1].positionRadius[0]=pair[1].positionRadius[2]=-7.90625f;check(pair,8,0.8f,0.03125f,nullptr,true);
    bad=fixture(3);bad[0].positionRadius[1]=0.03125f;check(bad,1,0.8f,0.03125f,nullptr,true);
    bad[0].velocityInvMass[3]=0;check(bad,1,0.8f,0.03125f,nullptr,true);
    GpuJacobiContacts recovering(assets,3);
    bad=fixture(3);bad[0].positionRadius[3]=0.05f;
    check(bad,1,0.8f,0.03125f,&recovering);check(fixture(3),1,0.8f,0.03125f,&recovering);
    {
        GpuJacobiContacts zero(assets,0);const ParticleGpu dummy{};
        ContactHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(&dummy,sizeof(dummy)),layout,BGFX_BUFFER_COMPUTE_READ));
        ContactHandle image(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle staging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        zero.submit(0,state.value,materialBuffer.value,std::uint32_t(materials.size()));
        bgfx::setBuffer(0,zero.status(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::array<std::uint32_t,4> status{};bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;
        bgfx::blit(2,dst,src);const auto ready=bgfx::read(dst,status.data());while(bgfx::frame()<ready) SDL_PumpEvents();
        require(status==std::array<std::uint32_t,4>{0,0,0,0},"Zero-capacity Jacobi iteration failed");++cases;
    }
    auto empty=fixture(129);for(auto& p:empty) p.identity[1]=0;check(empty);
    std::cout<<"GPU Jacobi immutable-input corrections passed: "<<cases<<" fixtures, mass weighting, coincident pairs, bounds, failure rollback and multi-iteration rebuilds\n";
}
}
