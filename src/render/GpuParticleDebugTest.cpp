#include "GpuParticleDebug.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "core/ParticleTypes.hpp"
#include "core/Settling.hpp"
#include "core/SpatialGrid.hpp"
#include "assets/Materials.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct Handle {
    T value;explicit Handle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Debug test resource unavailable");}
    ~Handle(){bgfx::destroy(value);}Handle(const Handle&)=delete;
};
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
}
void runGpuParticleDebugTest(const std::filesystem::path& assets) {
    GpuParticleDebug debug(assets);unsigned cases=0;
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    std::array<MaterialGpu,2> materials{};
    materials[0].color[0]=1;materials[0].color[3]=1;materials[0].identity[2]=7;
    materials[1].color[1]=1;materials[1].color[3]=1;materials[1].identity[2]=9;
    Handle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),lanes,BGFX_BUFFER_COMPUTE_READ));
    Handle exporter(loadCompute(assets,"cs_export"));Handle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    SpatialGrid grid({0,0,0},0.125f,{-64,0,-64},{128,128,128});
    for(unsigned count:{1u,127u,128u,129u,255u,256u,257u}) {
        std::vector<ParticleGpu> particles(count);std::vector<SettlingState> history(count);
        for(unsigned i=0;i<count;++i){
            auto& p=particles[i];p={{-0.125f+float(i%3)*0.125f,0.0625f,0.125f,0.0625f},{float(i%3)*6,0,0,1},{300,0,0,0},{i%2,1,i,3}};
            history[i]={3,0,0,0};
            switch(i%32){
                case 1:history[i].quietTicks=10;break;
                case 2:history[i]={3,30,0,1};break;
                case 3:history[i]={3,30,2,1};break;
                case 4:history[i]={2,30,0,1};break;
                case 5:p.identity[1]=0;break;
                case 6:p.identity[0]=2;break;
                case 7:p.positionRadius[0]=-std::numeric_limits<float>::denorm_min();break;
                case 8:p.positionRadius[0]=8;break;
                case 9:p.positionRadius[1]=-1;break;
                case 10:p.velocityInvMass[0]=std::numeric_limits<float>::max();break;
                case 11:p.velocityInvMass[0]=std::numeric_limits<float>::quiet_NaN();break;
                case 12:p.positionRadius[0]=std::numeric_limits<float>::quiet_NaN();break;
                case 13:p.positionRadius[3]=0;break;
                case 14:p.positionRadius[0]=std::nextafter(-0.125f,-1.0f);break;
                case 15:p.positionRadius[0]=std::nextafter(0.125f,0.0f);break;
                case 16:p.velocityInvMass[0]=-6;break;
                case 17:p.velocityInvMass[0]=0;p.velocityInvMass[1]=6;break;
                case 18:p.velocityInvMass[0]=0;p.velocityInvMass[1]=-6;break;
                case 19:p.velocityInvMass[0]=0;p.velocityInvMass[2]=6;break;
                case 20:p.velocityInvMass[0]=0;p.velocityInvMass[2]=-6;break;
                case 21:p.velocityInvMass[0]=3;p.velocityInvMass[1]=4;break;
                case 22:p.velocityInvMass[0]=-3;p.velocityInvMass[1]=-4;break;
                case 23:p.velocityInvMass[0]=std::numeric_limits<float>::max();p.velocityInvMass[1]=-p.velocityInvMass[0];p.velocityInvMass[2]=p.velocityInvMass[0];break;
                case 24:p.velocityInvMass[0]=std::numeric_limits<float>::min();break;
                case 25:p.velocityInvMass[0]=-std::numeric_limits<float>::denorm_min();break;
                case 26:p.velocityInvMass[0]=-0.0f;break;
                case 27:p.velocityInvMass[1]=std::numeric_limits<float>::infinity();break;
                case 28:p.velocityInvMass[2]=-std::numeric_limits<float>::infinity();break;
                case 29:p.velocityInvMass[0]=std::numeric_limits<float>::min();p.velocityInvMass[1]=p.velocityInvMass[0];break;
                case 30:p.velocityInvMass[0]=1;p.velocityInvMass[2]=-1;break;
                case 31:p.velocityInvMass[0]=-0.5f;p.velocityInvMass[1]=0.5f;p.velocityInvMass[2]=-0.5f;break;
            }
        }
        Handle state(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),count*sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
        Handle histories(bgfx::createDynamicVertexBuffer(bgfx::copy(history.data(),count*sizeof(SettlingState)),lanes,BGFX_BUFFER_COMPUTE_READ));
        Handle instances(bgfx::createDynamicVertexBuffer(count*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Handle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Handle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        // Reuse output without a simulation tick: every mode must restore all
        // presentation fields, including returning to material colors while paused.
        for(unsigned mode:{0u,1u,2u,3u,4u,0u,3u}) {
            const bool fault=cases%7==6;const std::array<std::uint32_t,4> stats{fault?1u:0u,17,23,11};
            Handle status(bgfx::createDynamicIndexBuffer(bgfx::copy(stats.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
            debug.submit(0,count,ParticleDebugMode(mode),state.value,materialBuffer.value,2,histories.value,status.value,instances.value);
            const float params[]{0,float(count),0,2};bgfx::setUniform(uniform.value,params);bgfx::setBuffer(0,instances.value,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            std::vector<std::array<std::uint32_t,16>> pixels(count);bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,pixels.data());while(bgfx::frame()<ready)SDL_PumpEvents();
            for(unsigned i=0;i<count;++i){const auto& p=particles[i];RenderInstance expected{};
                if((p.identity[1]&1) && p.identity[0]<2 && std::isfinite(p.positionRadius[0]) && p.positionRadius[3]>0) {
                    std::memcpy(expected.positionRadius,p.positionRadius,16);expected.visual[0]=p.identity[0]?0xff00ff00u:0xff0000ffu;
                    expected.visual[1]=materials[p.identity[0]].identity[2];expected.visual[2]=i;expected.visual[3]=1;
                    if(mode==1 || mode==4){
                        std::array<double,3> v{p.velocityInvMass[0],p.velocityInvMass[1],p.velocityInvMass[2]};
                        if(!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]))expected.visual[0]=0xffff00ffu;
                        else {
                            std::array<double,3> color{};
                            if(mode==1){double t=std::min(std::hypot(v[0],v[1],v[2])/12.0,1.0);color={32+223*t,96*(1-t),255*(1-t)};}
                            else {
                                for(auto& x:v)if(std::abs(x)<std::numeric_limits<float>::min())x=0;
                                double norm=std::hypot(v[0],v[1],v[2]);
                                for(unsigned axis=0;axis<3;++axis)color[axis]=127.5*(1+(norm?v[axis]/norm:0));
                            }
                            expected.visual[0]=0xff000000u;
                            for(unsigned axis=0;axis<3;++axis)expected.visual[0]|=std::uint32_t(std::floor(color[axis]+0.5))<<(8*axis);
                        }
                    }
                    if(mode==2){auto key=grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]});
                        if(!key)expected.visual[0]=0xffff00ffu;else {auto h=*key*1664525u+1013904223u;expected.visual[0]=(64u+(h&127u))|((64u+((h>>8)&127u))<<8)|((64u+((h>>16)&127u))<<16)|0xff000000u;}}
                    if(mode==3){const auto& h=history[i];expected.visual[0]=0xff808080u;if(fault)expected.visual[0]=0xffff00ffu;else if(h.generation==p.identity[3]){if(h.candidate)expected.visual[0]=h.unsupportedTicks?0xff2080ffu:0xff60dc40u;else if(h.quietTicks)expected.visual[0]=0xff20c8ffu;}}
                }
                if(std::memcmp(&expected,pixels[i].data(),sizeof(expected)))throw std::runtime_error("Debug render instance mismatch mode "+std::to_string(mode)+" slot "+std::to_string(i));
            }
            ++cases;
        }
        // Read original state and history after all presentation changes.
        for(unsigned which=0;which<2;++which){const float params[]{0,float(count),0,which==0?4.0f:1.0f};
            bgfx::setUniform(uniform.value,params);bgfx::setBuffer(0,which==0?state.value:histories.value,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            std::vector<std::array<std::uint32_t,16>> pixels(count);bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,pixels.data());while(bgfx::frame()<ready)SDL_PumpEvents();
            for(unsigned i=0;i<count;++i)require(std::memcmp(pixels[i].data(),which==0?static_cast<const void*>(&particles[i]):static_cast<const void*>(&history[i]),which==0?sizeof(ParticleGpu):sizeof(SettlingState))==0,"Debug view changed simulation/history bytes");
        }
    }
    std::cout<<"GPU particle debug passed: "<<cases<<" mode changes, full render instances, signed grid boundaries, stale generations, faults and immutable simulation/history\n";
}
}
