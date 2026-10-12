#include "core/ComputeDispatch.hpp"
#include "GpuLayoutTest.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include <SDL3/SDL.h>
#include <bgfx/bgfx.h>
#include <array>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
// Own handles through exception paths; bgfx queues destruction safely.
template<class T> struct Handle {
    T value = BGFX_INVALID_HANDLE;
    explicit Handle(T handle) : value(handle) {
        if(!bgfx::isValid(value)) throw std::runtime_error("GPU test resource creation failed");
    }
    ~Handle() { bgfx::destroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
}
void runGpuLayoutTest(const std::filesystem::path& assets) {
    const auto* caps=bgfx::getCaps();
    if(!(caps->formats[bgfx::TextureFormat::RGBA32U] & BGFX_CAPS_FORMAT_TEXTURE_IMAGE_WRITE))
        throw std::runtime_error("GPU layout validation requires RGBA32U image writes");
    Handle reset(loadCompute(assets,"cs_reset"));
    Handle exportState(loadCompute(assets,"cs_export"));
    Handle integrate(loadCompute(assets,"cs_particles"));
    Handle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    bgfx::VertexLayout lanes;
    lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    const auto materialLibrary=loadMaterials(assets/"materials"/"default.json");
    const auto materialRecords=materialLibrary.packed();
    Handle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materialRecords.data(),uint32_t(materialRecords.size()*sizeof(MaterialGpu))),lanes,BGFX_BUFFER_COMPUTE_READ));
    {
        const auto count=uint32_t(materialRecords.size());
        Handle output(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle staging(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<MaterialGpu> actual(count);
        const float parameters[]{0,float(count),float(materialRecords.size()),0};
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,materialBuffer.value,bgfx::Access::Read);
        bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(0,exportState.value,pw::dispatchGroups(count));
        bgfx::TextureRegion dst;dst.handle=staging.value;
        bgfx::TextureRegion src;src.handle=output.value;
        bgfx::blit(1,dst,src);
        const auto ready=bgfx::read(dst,actual.data());
        while(bgfx::frame()<ready) SDL_PumpEvents();
        if(std::memcmp(actual.data(),materialRecords.data(),count*sizeof(MaterialGpu))!=0)
            throw std::runtime_error("GPU material ABI round-trip mismatch");
        std::cout<<"GPU material ABI round-trip passed: "<<count<<" profiles\n";
    }
    // Exercise workgroup boundaries with exact, non-symmetric values in all lanes.
    for(uint32_t count : {1u,127u,128u,129u,255u,256u,257u}) {
        std::vector<ParticleGpu> expected(count), actual(count);
        for(uint32_t i=0;i<count;++i) {
            expected[i]={{float(i)+0.25f,-3.5f,7.125f,0.0625f},
                         {-1.25f,float(i)*0.5f,2.75f,i%3u==0u?0.0f:(i%3u==1u?-0.0f:0.125f)},
                         {293.25f,0.5f,0.0f,float(i)*0.25f},
                         {i%uint32_t(materialRecords.size()),0x12345679u,1000u+i,19u}};
        }
        Handle upload(bgfx::createDynamicVertexBuffer(bgfx::copy(expected.data(),uint32_t(expected.size()*sizeof(ParticleGpu))),lanes,BGFX_BUFFER_COMPUTE_READ));
        Handle state(bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle instances(bgfx::createDynamicVertexBuffer(count*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle integrated(bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle output(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle staging(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        const float parameters[]{0,float(count),float(materialRecords.size()),0};
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,upload.value,bgfx::Access::Read);
        bgfx::setBuffer(1,state.value,bgfx::Access::Write);
        bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer.value,bgfx::Access::Read);
        bgfx::dispatch(0,reset.value,pw::dispatchGroups(count));
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exportState.value,pw::dispatchGroups(count));
        bgfx::TextureRegion dst; dst.handle=staging.value;
        bgfx::TextureRegion src; src.handle=output.value;
        bgfx::blit(2,dst,src);
        const uint32_t ready=bgfx::read(dst,actual.data());
        while(bgfx::frame()<ready) SDL_PumpEvents();
        if(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(ParticleGpu))!=0)
            throw std::runtime_error("GPU particle ABI round-trip mismatch at count " + std::to_string(count));
        std::cout << "GPU ABI + ordered compute round-trip passed: " << count << " particles\n";
        const float dt=1.0f/60.0f;
        const float stepParameters[]{dt,float(count),float(materialRecords.size()),0};
        bgfx::setUniform(uniform.value,stepParameters);
        bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setBuffer(1,integrated.value,bgfx::Access::Write);
        bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materialBuffer.value,bgfx::Access::Read);
        bgfx::dispatch(3,integrate.value,pw::dispatchGroups(count));
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,integrated.value,bgfx::Access::Read);
        bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(4,exportState.value,pw::dispatchGroups(count));
        bgfx::blit(5,dst,src);
        const auto integratedReady=bgfx::read(dst,actual.data());
        while(bgfx::frame()<integratedReady) SDL_PumpEvents();
        for(uint32_t i=0;i<count;++i) {
            const auto& material=materialRecords[expected[i].identity[0]];
            auto& p=expected[i].positionRadius;
            auto& v=expected[i].velocityInvMass;
            if(v[3]==0.0f) {
                // Independent invariant: even outside the walls/floor, a static
                // body's complete pose and velocity must retain their bits.
                if(std::memcmp(p,actual[i].positionRadius,sizeof(expected[i].positionRadius))!=0 ||
                   std::memcmp(v,actual[i].velocityInvMass,sizeof(expected[i].velocityInvMass))!=0)
                    throw std::runtime_error("GPU integration moved a zero-inverse-mass body");
            } else {
                v[1]-=9.81f*dt;
                for(unsigned axis=0;axis<3;++axis) {v[axis]*=std::max(0.0f,1.0f-material.mechanical[3]*dt);p[axis]+=v[axis]*dt;}
                if(p[1]<p[3]) {p[1]=p[3];v[1]=std::abs(v[1])*material.mechanical[2];v[0]*=1.0f-material.mechanical[1];v[2]*=1.0f-material.mechanical[1];}
                for(unsigned axis : {0u,2u}) if(std::abs(p[axis])>8.0f-p[3]) {
                    p[axis]=std::copysign(8.0f-p[3],p[axis]); v[axis]*=-material.mechanical[2];
                }
            }
            expected[i].thermalState[3]+=dt;
            for(unsigned lane=0;lane<3;++lane) {
                const float* reference=lane==0?expected[i].positionRadius:lane==1?expected[i].velocityInvMass:expected[i].thermalState;
                const float* observed=lane==0?actual[i].positionRadius:lane==1?actual[i].velocityInvMass:actual[i].thermalState;
                for(unsigned field=0;field<4;++field)
                    if(!std::isfinite(observed[field]) || std::abs(reference[field]-observed[field])>1e-5f*std::max(1.0f,std::abs(reference[field])))
                        throw std::runtime_error("GPU integration differs from CPU reference");
            }
            if(std::memcmp(expected[i].identity,actual[i].identity,sizeof(expected[i].identity))!=0)
                throw std::runtime_error("GPU integration corrupted particle identity");
        }
        std::cout << "GPU gravity/contact and static-body invariant check passed: " << count << " particles\n";
        Handle renderOutput(bgfx::createTexture2D(2,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle renderStaging(bgfx::createTexture2D(2,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<RenderInstance> rendered(count);
        const float renderParameters[]{0,float(count),0,2};
        bgfx::setUniform(uniform.value,renderParameters);
        bgfx::setBuffer(0,instances.value,bgfx::Access::Read);
        bgfx::setImage(1,renderOutput.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(6,exportState.value,pw::dispatchGroups(count));
        bgfx::TextureRegion renderDst;renderDst.handle=renderStaging.value;
        bgfx::TextureRegion renderSrc;renderSrc.handle=renderOutput.value;
        bgfx::blit(7,renderDst,renderSrc);
        const auto renderReady=bgfx::read(renderDst,rendered.data());
        while(bgfx::frame()<renderReady) SDL_PumpEvents();
        for(uint32_t i=0;i<count;++i) {
            const auto& material=materialRecords[expected[i].identity[0]];
            uint32_t color=0;
            for(unsigned c=0;c<4;++c) color|=uint32_t(material.color[c]*255.0f+0.5f)<<(8*c);
            if(std::memcmp(actual[i].positionRadius,rendered[i].positionRadius,sizeof(rendered[i].positionRadius))!=0 ||
               rendered[i].visual[0]!=color || rendered[i].visual[1]!=material.identity[2] ||
               rendered[i].visual[2]!=expected[i].identity[2] || rendered[i].visual[3]!=expected[i].identity[1])
                throw std::runtime_error("GPU render ABI/material classification mismatch");
        }
        std::cout<<"GPU render records/color/class/IDs passed: "<<count<<" particles\n";


    }
}
}
