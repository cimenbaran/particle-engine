#include "core/ComputeDispatch.hpp"
#include "GpuVisibilityTest.hpp"
#include "ParticleVisibility.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct Owned {
    T value;
    explicit Owned(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Visibility test resource failed");}
    ~Owned(){bgfx::destroy(value);}
    Owned(const Owned&)=delete;
};
void require(bool condition,const char* message){if(!condition) throw std::runtime_error(message);}
}
void runGpuVisibilityTest(const std::filesystem::path& assets) {
    Owned exporter(loadCompute(assets,"cs_export"));
    Owned argsExporter(loadCompute(assets,"cs_indirect_export"));
    Owned sim(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).add(bgfx::Attrib::TexCoord1,4,bgfx::AttribType::Float).end();
    const float identity[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    const auto planes=extractFrustum(identity,true);
    for(bool allowIndirect:{false,true}) for(uint32_t capacity:{1u,127u,128u,129u,255u,256u,257u,16384u}) {
        ParticleVisibility visibility(assets,capacity,allowIndirect);
        PoolDiagnostics diagnostic(assets);
        Owned output(bgfx::createTexture2D(2,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned staging(bgfx::createTexture2D(2,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        Owned argsOutput(bgfx::createTexture2D(2,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned argsStaging(bgfx::createTexture2D(2,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        // Empty, tangent, all culled, disabled, full, and zero-to-one depth.
        for(unsigned scenario=0;scenario<6;++scenario) {
            std::vector<RenderInstance> source(capacity),actual(capacity);
            std::vector<uint32_t> active(capacity,0xffffffffu);
            std::vector<bool> expected(capacity),seen(capacity);
            uint32_t activeCount=0,visibleCount=0;
            for(uint32_t i=0;i<capacity;++i) {
                const bool alive=scenario==4 || (scenario!=0 && i%5!=4);
                const float x=scenario>=4?0.0f:scenario==2?3.0f:(i%4==0?0.0f:i%4==1?1.125f:i%4==2?-1.125f:2.0f);
                const float z=scenario==5?(i%4==0?-0.125f:i%4==1?-0.25f:i%4==2?1.125f:1.25f):0.0f;
                source[i]={{x,0,z,0.125f},{0xff123400u+i,i%3,i,alive?1u:0u}};
                if(alive) active[activeCount++]=i;
                // Exact tangent spheres remain visible in both depth conventions.
                expected[i]=alive && (scenario==4 || (scenario==5?i%2==0:(scenario==3 || (scenario!=2 && i%4!=3))));
                if(expected[i]) ++visibleCount;
            }
            Owned input(bgfx::createDynamicVertexBuffer(bgfx::copy(source.data(),capacity*sizeof(RenderInstance)),layout,BGFX_BUFFER_COMPUTE_READ));
            Owned activeBuffer(bgfx::createDynamicIndexBuffer(bgfx::copy(active.data(),capacity*sizeof(uint32_t)),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
            const uint32_t poolCounts[]{activeCount,capacity-activeCount,0,0};
            Owned poolBuffer(bgfx::createDynamicIndexBuffer(bgfx::copy(poolCounts,sizeof(poolCounts)),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
            visibility.prepare(0,input.value,activeBuffer.value,poolBuffer.value,scenario==5?extractFrustum(identity,false):planes,scenario!=3);
            const float params[]{0,float(capacity),0,2};
            bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
            bgfx::setUniform(sim.value,params);bgfx::setBuffer(0,visibility.instances(),bgfx::Access::Read);
            bgfx::setImage(1,output.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,pw::dispatchGroups(capacity));
            std::array<uint32_t,8> arguments{};
            if(visibility.supportsIndirect()) {
                // bgfx indirect buffers have a UAV but no D3D11 SRV.
                bgfx::setBuffer(0,visibility.arguments(),bgfx::Access::ReadWrite);
                bgfx::setImage(1,argsOutput.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
                bgfx::dispatch(1,argsExporter.value,1);
            }
            require(diagnostic.request(1,visibility.counts(),scenario),"Visibility diagnostic ring full");
            bgfx::TextureRegion dst;dst.handle=staging.value;bgfx::TextureRegion src;src.handle=output.value;
            bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
            if(visibility.supportsIndirect()) {
                dst.handle=argsStaging.value;src.handle=argsOutput.value;
                bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,arguments.data()));
            }
            uint32_t frame;do {frame=bgfx::frame();diagnostic.poll(frame);SDL_PumpEvents();} while(frame<ready || diagnostic.pending());
            require(diagnostic.latest().valid && diagnostic.latest().active==visibleCount,"GPU visible count mismatch");
            const RenderInstance zero{};
            for(uint32_t i=0;i<capacity;++i) {
                if(i<visibleCount) {
                    const auto id=actual[i].visual[2];
                    require(id<capacity && expected[id] && !seen[id],"Unexpected/duplicate compacted instance");
                    require(std::memcmp(&actual[i],&source[id],sizeof(RenderInstance))==0,"Compaction changed instance metadata");
                    seen[id]=true;
                } else require(std::memcmp(&actual[i],&zero,sizeof(zero))==0,"Fallback tail contains stale instances");
            }
            if(visibility.supportsIndirect()) {
                const std::array<uint32_t,8> expectedArguments{6,visibleCount,0,0,0,0,0,0};
                require(arguments==expectedArguments,"GPU indirect draw arguments mismatch");
            }
        }
        std::cout<<"GPU frustum/compaction/"<<(visibility.supportsIndirect()?"indirect":"masked fallback")<<" passed: "<<capacity<<" slots\n";
    }
}
}
