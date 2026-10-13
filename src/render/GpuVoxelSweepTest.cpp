#include "GpuVoxelSweep.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T>struct VoxelHandle {
    T value;explicit VoxelHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Voxel test allocation failed");}
    ~VoxelHandle(){bgfx::destroy(value);}VoxelHandle(const VoxelHandle&)=delete;
};
struct Segment {std::array<float,4> start,end;};
static_assert(sizeof(Segment)==32);
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
}
void runGpuVoxelSweepTest(const std::filesystem::path& assets) {
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    VoxelHandle exporter(loadCompute(assets,"cs_export"));
    VoxelHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned batches=0,queries=0;
    auto check=[&](std::vector<Segment> segments,std::array<float,3> lower,GpuVoxelSweep* reuse=nullptr) {
        const auto count=std::uint32_t(segments.size());require(count>0 && count<=4096,"Invalid voxel test size");
        auto owned=reuse?nullptr:std::make_unique<GpuVoxelSweep>(assets,count);auto& query=reuse?*reuse:*owned;
        VoxelHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(segments.data(),count*sizeof(Segment)),lanes,BGFX_BUFFER_COMPUTE_READ));
        query.submit(0,input.value,lower);
        VoxelHandle image(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VoxelHandle staging(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        VoxelHandle originalImage(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        VoxelHandle originalStaging(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        for(auto pair:{std::pair{query.output(),image.value},std::pair{input.value,originalImage.value}}) {
            const float p[]{0,float(count),0,2};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);
            bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        std::vector<std::array<float,8>> actual(count);std::vector<Segment> original(count);
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=originalImage.value;dst.handle=originalStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
        while(bgfx::frame()<ready)SDL_PumpEvents();
        require(std::memcmp(original.data(),segments.data(),count*sizeof(Segment))==0,"Voxel query mutated input");
        for(unsigned i=0;i<count;++i) {
            const auto& inputRow=segments[i];const auto& row=actual[i];const auto kind=std::bit_cast<std::uint32_t>(row[7]);
            bool valid=inputRow.start[3]==0.0625f && inputRow.end[3]==0;
            reference::VoxelVector start,end,corner;
            for(unsigned axis=0;axis<3;++axis) {
                start[axis]=inputRow.start[axis];end[axis]=inputRow.end[axis];corner[axis]=lower[axis];
                valid&=std::isfinite(start[axis]) && std::isfinite(end[axis]) && std::abs(start[axis])<=64 && std::abs(end[axis])<=64;
            }
            for(unsigned lane=0;lane<7;++lane)require(std::isfinite(row[lane]),"Nonfinite voxel query output");
            if(!valid){require(kind==3,"Voxel query accepted malformed segment");continue;}
            auto expected=reference::sweepVoxelSphere(start,end,corner);
            const unsigned expectedKind=expected.initialOverlap?2:expected.hit?1:0;
            if(kind!=expectedKind)throw std::runtime_error("Voxel kind mismatch batch "+std::to_string(batches)+" slot "+std::to_string(i)+" actual "+std::to_string(kind)+" expected "+std::to_string(expectedKind));
            if(std::abs(row[3]-expected.time)>2e-5)throw std::runtime_error("Voxel time mismatch slot "+std::to_string(i));
            for(unsigned axis=0;axis<3;++axis) {
                if(std::abs(row[axis]-expected.normal[axis])>5e-4)throw std::runtime_error("Voxel normal mismatch slot "+std::to_string(i));
                if(std::abs(row[axis+4]-expected.centre[axis])>2e-5)throw std::runtime_error("Voxel centre mismatch slot "+std::to_string(i));
            }
        }
        ++batches;queries+=count;
    };
    std::vector<Segment> special;
    auto add=[&](std::array<float,3> start,std::array<float,3> end) {
        special.push_back({{start[0],start[1],start[2],0.0625f},{end[0],end[1],end[2],0}});
    };
    for(unsigned axis=0;axis<3;++axis)for(float sign:{-1.0f,1.0f}) {
        std::array<float,3> start{0.0625f,0.0625f,0.0625f},end=start;start[axis]=sign;end[axis]=-sign;add(start,end);
    }
    add({-1,-0.0625f*0.6f,0.0625f},{1,-0.0625f*0.6f,0.0625f});
    add({-1,-0.0625f/3,-0.0625f/3},{1,-0.0625f/3,-0.0625f/3});
    add({-1,-0.0625f*0.9f,-0.0625f*0.9f},{1,-0.0625f*0.9f,-0.0625f*0.9f});
    for(float y:{-0.0625f,std::nextafter(-0.0625f,-1.0f),std::nextafter(-0.0625f,0.0f)})add({-1,y,0.0625f},{1,y,0.0625f});
    add({-1,0.0625f,0.0625f},{-0.0625f,0.0625f,0.0625f});
    add({-1,0.0625f,0.0625f},{std::nextafter(-0.0625f,-1.0f),0.0625f,0.0625f});
    add({-0.0625f,0.0625f,0.0625f},{-1,0.0625f,0.0625f});
    add({0.0625f,0.0625f,0.0625f},{0.0625f,0.0625f,0.0625f});
    add({1,1,1},{1,1,1});add({-64,0.0625f,0.0625f},{64,0.0625f,0.0625f});
    auto invalid=special.front();invalid.start[0]=std::numeric_limits<float>::quiet_NaN();special.push_back(invalid);
    invalid=special.front();invalid.start[3]=0;special.push_back(invalid);
    invalid=special.front();invalid.end[3]=1;special.push_back(invalid);
    invalid=special.front();invalid.end[0]=65;special.push_back(invalid);
    check(special,{});
    std::mt19937 random(187721);auto coordinate=[&]{return float(random()%1000001)/1000000.0f-0.5f;};
    std::vector<Segment> many=special;
    while(many.size()<4096)many.push_back({{coordinate(),coordinate(),coordinate(),0.0625f},{coordinate(),coordinate(),coordinate(),0}});
    for(unsigned count:{1u,127u,128u,129u,255u,256u,257u,4096u})check({many.begin(),many.begin()+count},{});
    auto translated=many;for(auto& segment:translated)for(unsigned axis=0;axis<3;++axis){const float offset=axis==0?-4:axis==1?8:-12;segment.start[axis]+=offset;segment.end[axis]+=offset;}
    check(translated,{-4,8,-12});
    GpuVoxelSweep reused(assets,257);check({many.begin(),many.begin()+257},{},&reused);check({translated.begin(),translated.begin()+257},{-4,8,-12},&reused);
    bool rejected=false;try{reused.submit(0,reused.output(),{});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Voxel query accepted output alias");
    VoxelHandle dummy(bgfx::createDynamicVertexBuffer(bgfx::copy(special.data(),sizeof(Segment)),lanes,BGFX_BUFFER_COMPUTE_READ));
    rejected=false;try{reused.submit(0,dummy.value,{64,0,0});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Voxel query accepted outside corner");
    GpuVoxelSweep empty(assets,0);empty.submit(0,dummy.value,{});bgfx::frame();
    require(GpuVoxelSweep::payloadBytes(0)==32 && GpuVoxelSweep::payloadBytes(257)==8224,"Voxel query payload mismatch");
    std::cout<<"GPU voxel sweep passed: "<<batches<<" batches, "<<queries<<" queries, CPU geometry agreement, invalid input, reuse and immutable segments\n";
}
}
