#include "GpuLayoutTest.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
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
    // Exercise workgroup boundaries with exact, non-symmetric values in all lanes.
    for(uint32_t count : {1u,127u,128u,129u}) {
        std::vector<ParticleGpu> expected(count), actual(count);
        for(uint32_t i=0;i<count;++i) {
            expected[i]={{float(i)+0.25f,-3.5f,7.125f,0.0625f},
                         {-1.25f,float(i)*0.5f,2.75f,0.125f},
                         {293.25f,0.5f,0.75f,float(i)*0.25f},
                         {5u,0x12345678u,1000u+i,19u}};
        }
        Handle upload(bgfx::createDynamicVertexBuffer(bgfx::copy(expected.data(),uint32_t(expected.size()*sizeof(ParticleGpu))),lanes,BGFX_BUFFER_COMPUTE_READ));
        Handle state(bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle instances(bgfx::createDynamicVertexBuffer(count*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle integrated(bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle output(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle staging(bgfx::createTexture2D(4,uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        const float parameters[]{0,float(count),0,0};
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,upload.value,bgfx::Access::Read);
        bgfx::setBuffer(1,state.value,bgfx::Access::Write);
        bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::dispatch(0,reset.value,(count+127)/128);
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exportState.value,(count+127)/128);
        bgfx::TextureRegion dst; dst.handle=staging.value;
        bgfx::TextureRegion src; src.handle=output.value;
        bgfx::blit(2,dst,src);
        const uint32_t ready=bgfx::read(dst,actual.data());
        while(bgfx::frame()<ready) SDL_PumpEvents();
        if(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(ParticleGpu))!=0)
            throw std::runtime_error("GPU particle ABI round-trip mismatch at count " + std::to_string(count));
        std::cout << "GPU ABI + ordered compute round-trip passed: " << count << " particles\n";
        const float dt=1.0f/60.0f;
        const float stepParameters[]{dt,float(count),0,0};
        bgfx::setUniform(uniform.value,stepParameters);
        bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setBuffer(1,integrated.value,bgfx::Access::Write);
        bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::dispatch(3,integrate.value,(count+127)/128);
        bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,integrated.value,bgfx::Access::Read);
        bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(4,exportState.value,(count+127)/128);
        bgfx::blit(5,dst,src);
        const auto integratedReady=bgfx::read(dst,actual.data());
        while(bgfx::frame()<integratedReady) SDL_PumpEvents();
        for(uint32_t i=0;i<count;++i) {
            auto& p=expected[i].positionRadius;
            auto& v=expected[i].velocityInvMass;
            v[1]-=9.81f*dt;
            for(unsigned axis=0;axis<3;++axis) {v[axis]*=1.0f-0.05f*dt;p[axis]+=v[axis]*dt;}
            if(p[1]<p[3]) {p[1]=p[3];v[1]=std::abs(v[1])*0.55f;v[0]*=0.98f;v[2]*=0.98f;}
            for(unsigned axis : {0u,2u}) if(std::abs(p[axis])>8.0f-p[3]) {
                p[axis]=std::copysign(8.0f-p[3],p[axis]); v[axis]*=-0.6f;
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
        std::cout << "GPU gravity/contact CPU-reference check passed: " << count << " particles\n";

    }
}
}
