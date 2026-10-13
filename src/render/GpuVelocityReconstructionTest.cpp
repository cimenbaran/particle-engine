#include "GpuVelocityReconstruction.hpp"
#include "GpuJacobiContacts.hpp"
#include "GpuSweptBounds.hpp"
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
#include <stdexcept>
#include <vector>
#include <utility>
namespace pw {
namespace {
template<class T> struct VelocityHandle {
    T value;
    explicit VelocityHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Velocity test allocation failed");}
    ~VelocityHandle(){bgfx::destroy(value);}
    VelocityHandle(const VelocityHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
}
void runGpuVelocityReconstructionTest(const std::filesystem::path& assets) {
    std::array<MaterialGpu,5> materials{};for(unsigned i=0;i<5;++i) {materials[i].identity[0]=i;materials[i].identity[1]=i==3?0:i==4?99:i;materials[i].mechanical[2]=0.5f;}
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    VelocityHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    VelocityHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    VelocityHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> before,std::vector<ParticleGpu> corrected,float dt,std::uint32_t upstream=0,std::uint32_t error=0,
                  GpuVelocityReconstruction* reused=nullptr,bool throughContacts=false,bool correctionMode=false,bool throughSweep=false) {
        const auto count=std::uint32_t(before.size()),storage=std::max(1u,count);require(corrected.size()==before.size(),"Velocity fixture sizes differ");
        auto owned=reused?nullptr:std::make_unique<GpuVelocityReconstruction>(assets,count);auto& velocity=reused?*reused:*owned;
        auto original=before,uploaded=corrected;if(!count) {original.resize(1);uploaded.resize(1);}
        VelocityHandle previous(bgfx::createDynamicVertexBuffer(bgfx::copy(original.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        VelocityHandle predicted(bgfx::createDynamicVertexBuffer(bgfx::copy(uploaded.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        const std::array<std::uint32_t,4> statusInput{upstream,0,0,0};
        VelocityHandle sourceStatus(bgfx::createDynamicIndexBuffer(bgfx::copy(statusInput.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        std::unique_ptr<GpuJacobiContacts> contacts;auto current=predicted.value;auto status=sourceStatus.value;
        if(throughContacts) {
            require(count==2,"Contact/velocity fixture requires two particles");contacts=std::make_unique<GpuJacobiContacts>(assets,count);
            contacts->submit(0,current,materialBuffer.value,std::uint32_t(materials.size()));current=contacts->output();status=contacts->status();
            // Equal masses, 0.0625 separation: each gets 0.025 m after relaxation.
            corrected[0].positionRadius[0]-=0.025f;corrected[1].positionRadius[0]+=0.025f;
        }
        auto baseReference=before;auto reconstructionBase=previous.value;
        std::unique_ptr<GpuSweptBounds> sweep;
        if(throughSweep) {
            require(count==1 && correctionMode && !throughContacts,"Invalid swept reconstruction fixture");
            sweep=std::make_unique<GpuSweptBounds>(assets,count);
            sweep->submit(0,previous.value,materialBuffer.value,std::uint32_t(materials.size()),status,dt,1000);
            current=sweep->output();status=sweep->status();reconstructionBase=current;
            // Unit drop at 10 m/s, e=0.5, dt=0.2: floor hit at t=0.1,
            // then 0.5 m upward. Keep +5 m/s, not the -2.5 m/s chord velocity.
            corrected=before;corrected[0].positionRadius[1]=0.5625f;corrected[0].velocityInvMass[1]=5;
            baseReference=corrected;
        }
        velocity.submit(0,reconstructionBase,current,materialBuffer.value,std::uint32_t(materials.size()),status,dt,
            correctionMode?VelocityReconstructionMode::Correction:VelocityReconstructionMode::Displacement);
        VelocityHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VelocityHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        VelocityHandle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VelocityHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        VelocityHandle newImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VelocityHandle newStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        VelocityHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VelocityHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count) for(auto pair:{std::pair{velocity.output(),image.value},std::pair{previous.value,oldImage.value},std::pair{current,newImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);
            bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,velocity.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(storage),oldActual(storage),newActual(storage);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count) {
            src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));
            src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,oldActual.data()));
            src.handle=newImage.value;dst.handle=newStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,newActual.data()));
        }
        while(bgfx::frame()<ready) SDL_PumpEvents();
        require(stats[0]==error,"Velocity reconstruction error mismatch");
        if(upstream) require(stats[1]==0 && stats[2]==0 && stats[3]==0,"Velocity consumed failed contact output");
        else {require(stats[1]+stats[2]+stats[3]==count,"Velocity slot accounting mismatch");require(stats[3]==(error?1u:0u),"Velocity invalid count mismatch");}
        unsigned updated=0,excluded=0;
        for(std::uint32_t i=0;i<count;++i) {
            require(std::memcmp(&oldActual[i],&before[i],sizeof(ParticleGpu))==0,"Velocity reconstruction mutated previous state");
            if(!throughContacts && !throughSweep) require(std::memcmp(&newActual[i],&corrected[i],sizeof(ParticleGpu))==0,"Velocity reconstruction mutated corrected input");
            else for(unsigned axis=0;axis<3;++axis) require(std::abs(newActual[i].positionRadius[axis]-corrected[i].positionRadius[axis])<1e-6,"Contact/velocity chain position mismatch");
            const auto& source=newActual[i];const bool dynamic=(source.identity[1]&1) && (source.identity[0]==0 || source.identity[0]==3) && source.velocityInvMass[3]!=0;
            if(dynamic) ++updated;else ++excluded;
            if(error || !dynamic) {require(std::memcmp(&actual[i],&source,sizeof(ParticleGpu))==0,"Velocity failure/exclusion changed corrected state");continue;}
            require(std::memcmp(actual[i].positionRadius,source.positionRadius,16)==0,"Velocity reconstruction moved positions/radius");
            require(std::memcmp(&actual[i].velocityInvMass[3],&source.velocityInvMass[3],sizeof(ParticleGpu)-7*sizeof(float))==0,"Velocity reconstruction changed mass, thermal state or identity");
            for(unsigned axis=0;axis<3;++axis) {
                const double expected=(double(corrected[i].positionRadius[axis])-baseReference[i].positionRadius[axis])/dt+
                    (correctionMode?baseReference[i].velocityInvMass[axis]:0.0);
                require(std::isfinite(actual[i].velocityInvMass[axis]) && std::abs(actual[i].velocityInvMass[axis]-expected)<=std::max(1e-9,std::abs(expected)*2e-5),"Reconstructed velocity differs from displacement/timestep");
            }
        }
        if(!error) require(stats[1]==updated && stats[2]==excluded,"Velocity update/exclusion count mismatch");
        ++cases;
    };
    auto fixture=[](std::uint32_t count) {
        std::vector<ParticleGpu> p(count);
        for(std::uint32_t i=0;i<count;++i) {
            p[i].positionRadius[0]=float(i%16)/16;p[i].positionRadius[1]=4;p[i].positionRadius[2]=-0.5f;
            p[i].positionRadius[3]=0.0625f;p[i].velocityInvMass[0]=0.125f;p[i].velocityInvMass[1]=0.25f;p[i].velocityInvMass[3]=1;
            p[i].thermalState[0]=300;p[i].identity[0]=i%2?3:0;p[i].identity[1]=1;p[i].identity[2]=i;p[i].identity[3]=7;
        }return p;
    };
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u}) {
        auto a=fixture(count),b=a;
        for(std::uint32_t i=0;i<count;++i) {
            b[i].positionRadius[0]+=0.03125f;b[i].positionRadius[1]-=0.0625f;b[i].positionRadius[2]+=0.015625f;b[i].thermalState[3]=0.125f;
            if(i%7==1) {a[i].identity[0]=b[i].identity[0]=1;}
            if(i%7==2) {a[i].identity[0]=b[i].identity[0]=2;}
            if(i%7==3) {a[i].identity[1]=b[i].identity[1]=0;b[i].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();}
            if(i%7==4) a[i].velocityInvMass[3]=b[i].velocityInvMass[3]=0;
        }
        check(a,b,0.125f);check(a,b,1.f/60);check(a,a,0.125f);
        check(a,b,0.125f,0,0,nullptr,false,true);check(a,a,0.125f,0,0,nullptr,false,true);
    }
    auto a=fixture(2),b=a;b[1].positionRadius[0]=a[1].positionRadius[0]=0.0625f;
    for(auto& p:b) {p.positionRadius[0]+=0.015625f;p.thermalState[3]=0.125f;}
    check(a,b,0.125f,0,0,nullptr,true);
    check(a,b,0.125f,0,0,nullptr,true,true);
    for(std::uint32_t generation:{16777216u,2147483648u,UINT32_MAX-1}) {
        auto previous=fixture(2);previous[0].identity[3]=generation;auto current=previous;
        check(previous,current,.125f);++current[0].identity[3];check(previous,current,.125f,0,2);
    }
    auto good=fixture(3),bad=good;bad[0].identity[3]++;check(good,bad,0.125f,0,2);
    bad=good;bad[0].identity[0]=3;check(good,bad,0.125f,0,2);
    bad=good;bad[0].positionRadius[3]=0.05f;check(good,bad,0.125f,0,2);
    bad=good;bad[0].velocityInvMass[3]=2;check(good,bad,0.125f,0,2);
    bad=good;bad[0].identity[2]=2;check(good,bad,0.125f,0,2);
    bad=good;bad[0].identity[0]=99;check(good,bad,0.125f,0,4);
    bad=good;bad[0].identity[0]=4;check(good,bad,0.125f,0,4);
    bad=good;bad[0].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(good,bad,0.125f,0,4);
    bad=good;bad[0].positionRadius[0]=std::numeric_limits<float>::infinity();check(bad,good,0.125f,0,4);
    bad=good;bad[0].velocityInvMass[3]=-1;check(bad,bad,0.125f,0,4);
    check(good,good,0.125f,4,1);
    auto one=fixture(1),moved=one;moved[0].positionRadius[0]=7;one[0].positionRadius[0]=-7;
    check(one,moved,std::numeric_limits<float>::min(),0,8);
    one=fixture(1);moved=one;moved[0].positionRadius[0]=std::numeric_limits<float>::min();check(one,moved,std::numeric_limits<float>::min());
    moved[0].positionRadius[0]=std::numeric_limits<float>::denorm_min();check(one,moved,std::numeric_limits<float>::min());
    moved[0].positionRadius[0]=-std::numeric_limits<float>::denorm_min();check(one,moved,std::numeric_limits<float>::min());
    one[0].positionRadius[0]=std::numeric_limits<float>::min();moved[0].positionRadius[0]=std::nextafter(one[0].positionRadius[0],1.f);check(one,moved,std::numeric_limits<float>::min());
    one=fixture(1);one[0].positionRadius[1]=1.0625f;one[0].velocityInvMass[0]=one[0].velocityInvMass[2]=0;one[0].velocityInvMass[1]=-10;
    check(one,one,0.2f,0,0,nullptr,false,true,true);
    one=fixture(1);moved=one;one[0].velocityInvMass[0]=std::numeric_limits<float>::quiet_NaN();
    check(one,moved,0.125f,0,8,nullptr,false,true);
    one=fixture(1);moved=one;one[0].velocityInvMass[0]=std::numeric_limits<float>::max();moved[0].positionRadius[0]=7;
    check(one,moved,std::numeric_limits<float>::min(),0,8,nullptr,false,true);
    check(good,good,0.125f,4,1,nullptr,false,true);
    GpuVelocityReconstruction recovery(assets,3);bad=good;bad[0].identity[3]++;
    check(good,bad,0.125f,0,2,&recovery);check(good,good,0.125f,0,0,&recovery);
    std::cout<<"GPU velocity reconstruction passed: "<<cases<<" fixtures, immutable inputs, contact chaining, exclusion, generation guards and whole-pass rollback\n";
}
}
