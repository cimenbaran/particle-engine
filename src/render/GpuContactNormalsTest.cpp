#include "GpuContactNormals.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include "../../tests/reference/ContactNormal.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct NormalHandle {
    T value;explicit NormalHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Normal fixture resource unavailable");}
    ~NormalHandle(){bgfx::destroy(value);}NormalHandle(const NormalHandle&)=delete;
};
struct NormalRecord {float directionGap[4];std::uint32_t metadata[4];};
static_assert(sizeof(NormalRecord)==32);
}
void runGpuContactNormalsTest(const std::filesystem::path& assets) {
    unsigned cases=0;
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,2> materials{};materials[1].identity[1]=1;
    NormalHandle material(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),lanes,BGFX_BUFFER_COMPUTE_READ));
    NormalHandle exporter(loadCompute(assets,"cs_export"));NormalHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    auto fixture=[](unsigned count) {
        std::vector<ParticleGpu> bodies(count);
        for(unsigned i=0;i<count;++i)bodies[i]={{float(i%16)*0.25f-2,4+float(i/16)*0.25f,0,0.0625f},{0,0,0,1},{300,0,0,0},{0,1,i,1}};
        return bodies;
    };
    auto check=[&](const std::vector<ParticleGpu>& bodies,AnalyticSphere sphere={},unsigned error=0,GpuContactNormals* reuse=nullptr) {
        const unsigned count=unsigned(bodies.size()),storage=std::max(1u,count);
        GpuContactNormals own(assets,count);auto& normals=reuse?*reuse:own;
        std::vector<ParticleGpu> input=bodies;if(input.empty())input.resize(1);
        NormalHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),storage*sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
        NormalHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        NormalHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        normals.submit(0,state.value,material.value,2,sphere);
        auto read=[&](bgfx::DynamicVertexBufferHandle buffer,unsigned laneCount) {
            const float p[]{0,float(count),0,float(laneCount)};bgfx::setUniform(uniform.value,p);
            bgfx::setBuffer(0,buffer,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            std::vector<std::array<std::uint32_t,16>> pixels(storage);
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);
            auto ready=bgfx::read(dst,pixels.data());while(bgfx::frame()<ready)SDL_PumpEvents();return pixels;
        };
        if(!count){bgfx::frame();++cases;return;}
        auto pixels=read(normals.output(),2);
        std::vector<reference::NormalBody> referenceBodies(count);
        for(unsigned i=0;i<count;++i){for(unsigned a=0;a<3;++a)referenceBodies[i].position[a]=bodies[i].positionRadius[a];
            referenceBodies[i].alive=(bodies[i].identity[1]&1)!=0;referenceBodies[i].solid=bodies[i].identity[0]==0;referenceBodies[i].dynamic=bodies[i].velocityInvMass[3]>0;}
        for(unsigned i=0;i<count;++i) {
            NormalRecord actual;std::memcpy(&actual,pixels[i].data(),sizeof(actual));
            auto fail=[&](const char* reason){throw std::runtime_error(std::string("Contact normal ")+reason+" fixture "+std::to_string(cases)+" slot "+std::to_string(i));};
            if(actual.metadata[2]!=error || actual.metadata[3])fail("error metadata mismatch");
            if(error){if(actual.metadata[0] || actual.metadata[1] || actual.directionGap[0] || actual.directionGap[1] || actual.directionGap[2])fail("failed result not empty");continue;}
            const auto expected=reference::contactNormal(referenceBodies,i,sphere);
            if(actual.metadata[0]!=unsigned(expected.source) || actual.metadata[1]!=expected.feature)fail("selected feature mismatch");
            for(unsigned a=0;a<3;++a)if(!std::isfinite(actual.directionGap[a]) || std::abs(actual.directionGap[a]-expected.direction[a])>2e-5)fail("direction mismatch");
            if(expected.source!=reference::NormalSource::None && std::abs(actual.directionGap[3]-expected.gap)>2e-6)fail("gap mismatch");
        }
        pixels=read(state.value,4);
        for(unsigned i=0;i<count;++i)if(std::memcmp(pixels[i].data(),&bodies[i],sizeof(ParticleGpu)))throw std::runtime_error("Normal diagnostic mutated particles");
        ++cases;
    };
    for(unsigned count:{0u,1u,127u,128u,129u,255u,256u,257u}) {
        auto p=fixture(count);check(p);
        for(unsigned i=0;i<count;++i){p[i].positionRadius[1]=0.0625f;if(i%5==1)p[i].identity[1]=0;if(i%5==2)p[i].identity[0]=1;if(i%5==3)p[i].velocityInvMass[3]=0;}
        check(p);
    }
    for(unsigned axis=0;axis<3;++axis)for(unsigned side=0;side<2;++side) {
        auto p=fixture(1);p[0].positionRadius[axis]=axis==1?(side?15.9375f:0.0625f):(side?7.9375f:-7.9375f);check(p);
    }
    auto p=fixture(3);p[0].positionRadius[0]=0;p[1].positionRadius[0]=0.125f;p[2].positionRadius[0]=-0.125f;check(p);
    p[2].positionRadius[0]=-0.0625f;check(p);
    p.resize(2);p[0].positionRadius[0]=0;p[1].positionRadius[0]=0.126f;check(p);
    p[1].positionRadius[0]=0.128f;check(p);
    p[0].positionRadius[0]=-7.9375f;p[0].positionRadius[1]=0.0625f;check(p);

    p=fixture(257);for(auto& b:p){b.positionRadius[0]=0;b.positionRadius[1]=4;}check(p);
    AnalyticSphere sphere{{0,4,0},0.9375f};
    for(unsigned axis=0;axis<3;++axis)for(unsigned side=0;side<2;++side){p=fixture(1);for(unsigned a=0;a<3;++a)p[0].positionRadius[a]=sphere.centre[a];p[0].positionRadius[axis]+=side?1.0f:-1.0f;check(p,sphere);}
    for(unsigned mask=0;mask<8;++mask){p=fixture(1);for(unsigned a=0;a<3;++a)p[0].positionRadius[a]=sphere.centre[a]+(mask&(1u<<a)?1.0f:-1.0f)/std::sqrt(3.0f);check(p,sphere);}
    p=fixture(2);p[0].positionRadius[0]=1;p[1].positionRadius[0]=1.125f;check(p,sphere);
    p=fixture(2);p[1].positionRadius[0]=p[0].positionRadius[0]+0.0625f;p[1].positionRadius[3]=0.125f;check(p,{},2);
    GpuContactNormals reused(assets,1);
    p=fixture(1);p[0].positionRadius[0]=9;check(p,{},1,&reused);
    p=fixture(1);p[0].positionRadius[0]=0;check(p,sphere,2,&reused); // Sphere centre.
    p=fixture(1);p[0].positionRadius[3]=0.125f;check(p,{},2,&reused);
    p=fixture(1);p[0].velocityInvMass[3]=-1;check(p,{},2,&reused);
    p=fixture(1);p[0].positionRadius[1]=0.0625f;check(p,{},0,&reused);
    p[0].positionRadius[1]=4;check(p,{},0,&reused); // No stale normal after moving away.
    std::cout<<"GPU contact normals passed: "<<cases<<" fixtures, independent CPU features/directions/gaps, dense ties, failures/reuse and immutable input\n";
}
}
