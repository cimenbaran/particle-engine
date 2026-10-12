#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "Shaders.hpp"
#include "core/SpatialGrid.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "assets/Materials.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct KeyHandle {
    T value;
    explicit KeyHandle(T handle):value(handle) {if(!bgfx::isValid(value)) throw std::runtime_error("Key test resource unavailable");}
    ~KeyHandle() {bgfx::destroy(value);}
    KeyHandle(const KeyHandle&)=delete;
    KeyHandle& operator=(const KeyHandle&)=delete;
};
}
void runGpuSpatialKeysTest(const std::filesystem::path& assets) {
    const SpatialGrid grid({0,0,0},0.125f,{-64,0,-64},{128,128,128});
    std::array<MaterialGpu,4> materials{};
    for(unsigned i=0;i<4;++i) {materials[i].identity[0]=i;materials[i].identity[1]=i==3?99:i;}
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    KeyHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    KeyHandle exporter(loadCompute(assets,"cs_radix_export")),statusExporter(loadCompute(assets,"cs_pool_stats"));
    KeyHandle uniform(bgfx::createUniform("u_radix",bgfx::UniformType::Vec4));
    const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
    const float tiny=std::numeric_limits<float>::denorm_min();
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u,16385u,1048576u}) {
        GpuSpatialKeys keys(assets,count);GpuRadixSort sort(assets,count);
        // Reuse the same objects: an invalid mixed epoch followed by a valid
        // epoch must clear errors, excluded counts and all old record keys.
        for(unsigned mixed=1;;--mixed) {
            std::vector<ParticleGpu> particles(std::max(1u,count));
            std::vector<std::array<std::uint32_t,2>> expected(std::max(1u,count));
            std::array<std::uint32_t,4> expectedStatus{};
            for(std::uint32_t i=0;i<count;++i) {
                auto& p=particles[i];
                p.positionRadius[0]=(int(i%128)-64)*0.125f+0.03125f;
                p.positionRadius[1]=((i/128)%128)*0.125f+0.03125f;
                p.positionRadius[2]=(int((i/16384)%128)-64)*0.125f+0.03125f;
                p.positionRadius[3]=0.05f;p.identity[1]=1;p.identity[2]=i;p.identity[3]=1;
                if(mixed) switch(i%16) {
                    case 1:p.identity[1]=0;p.positionRadius[0]=nan;break;
                    case 2:p.identity[0]=1;break;
                    case 3:p.identity[0]=2;break;
                    case 4:p.identity[0]=4;break;
                    case 5:p.positionRadius[0]=i%32<16?nan:inf;break;
                    case 6:p.positionRadius[1]=i%32<16?16:std::numeric_limits<float>::max();break;
                    case 7:p.identity[2]=i+1;break;
                    case 8:p.identity[3]=0;break;
                    case 9:p.identity[0]=3;break;
                    case 10:p.positionRadius[0]=i%32<16?-tiny:tiny;break;
                    case 11:p.positionRadius[1]=i%32<16?-tiny:tiny;break;
                    case 12:p.positionRadius[0]=-8;p.positionRadius[1]=i%32<16?0.f:-0.f;p.positionRadius[2]=-8;break;
                    case 13:p.positionRadius[0]=8;break;
                    case 14:p.positionRadius[0]=std::nextafter(-8.f,-inf);break;
                    case 15:p.positionRadius[0]=std::nextafter(8.f,0.f);p.positionRadius[1]=std::nextafter(16.f,0.f);break;
                }
                expected[i]={UINT32_MAX,i};
                if(!(p.identity[1]&1)) {++expectedStatus[1];continue;}
                if(p.identity[0]>=materials.size() || materials[p.identity[0]].identity[1]>2) {++expectedStatus[2];continue;}
                if(materials[p.identity[0]].identity[1]!=0) {++expectedStatus[1];continue;}
                if(p.identity[2]!=i || !p.identity[3] || !std::isfinite(p.positionRadius[0]) ||
                    !std::isfinite(p.positionRadius[1]) || !std::isfinite(p.positionRadius[2])) {++expectedStatus[2];continue;}
                const auto key=grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]});
                if(!key) {++expectedStatus[3];continue;}
                expected[i][0]=*key;++expectedStatus[0];
            }
            auto sorted=expected;std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a[0]<b[0];});
            KeyHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),std::uint32_t(particles.size()*sizeof(ParticleGpu))),layout,BGFX_BUFFER_COMPUTE_READ));
            const auto height=std::uint16_t((std::max(1u,count)+255)/256);
            KeyHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
            KeyHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
            KeyHandle statusImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
            KeyHandle statusStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
            std::vector<std::array<std::uint32_t,4>> actual(std::size_t(height)*256);std::array<std::uint32_t,4> actualStatus{};
            keys.submit(0,state.value,materialBuffer.value,std::uint32_t(materials.size()));sort.submit(0,keys.records());
            const float parameters[]{float(count),0,0,0};bgfx::setUniform(uniform.value,parameters);
            bgfx::setBuffer(0,sort.output(),bgfx::Access::Read);bgfx::setBuffer(1,keys.records(),bgfx::Access::Read);
            bgfx::setImage(2,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,dispatchGroups(std::max(1u,count)));
            bgfx::setBuffer(0,keys.status(),bgfx::Access::Read);bgfx::setImage(1,statusImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,statusExporter.value,1);
            bgfx::TextureRegion source;source.handle=image.value;bgfx::TextureRegion destination;destination.handle=staging.value;
            bgfx::blit(2,destination,source);const auto ready=bgfx::read(destination,actual.data());
            source.handle=statusImage.value;destination.handle=statusStaging.value;
            bgfx::blit(2,destination,source);const auto statusReady=bgfx::read(destination,actualStatus.data());
            while(bgfx::frame()<std::max(ready,statusReady)) SDL_PumpEvents();
            if(actualStatus!=expectedStatus) throw std::runtime_error("GPU spatial status mismatch at count "+std::to_string(count));
            for(std::uint32_t i=0;i<std::max(1u,count);++i) {
                const std::array<std::uint32_t,4> wanted{sorted[i][0],sorted[i][1],expected[i][0],expected[i][1]};
                if(actual[i]!=wanted) throw std::runtime_error("GPU spatial key/sort mismatch at count "+std::to_string(count)+" slot "+std::to_string(i));
            }
            if(!mixed) break;
        }
        std::cout<<"GPU spatial keys + sort passed: "<<count<<" slots, mixed errors then valid rebuild\n";
    }
}
}
