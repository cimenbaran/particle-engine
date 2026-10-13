#include "GpuNeighborsTest.hpp"
#include "GpuSpatialKeys.hpp"
#include "GpuRadixSort.hpp"
#include "GpuCellRanges.hpp"
#include "GpuCellHash.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/SpatialGrid.hpp"
#include "core/ComputeDispatch.hpp"
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
template<class T> struct NeighborHandle {
    T value;
    explicit NeighborHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Neighbor test allocation failed");}
    ~NeighborHandle(){bgfx::destroy(value);}
    NeighborHandle(const NeighborHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
}
void runGpuNeighborsTest(const std::filesystem::path& assets) {
    std::array<MaterialGpu,4> materials{};
    for(unsigned i=0;i<4;++i) {materials[i].identity[0]=i;materials[i].identity[1]=i==3?0:i;}
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    NeighborHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    NeighborHandle exporter(loadCompute(assets,"cs_neighbors_export"));
    NeighborHandle uniform(bgfx::createUniform("u_neighbors",bgfx::UniformType::Vec4));
    unsigned cases=0;std::uint64_t queriesChecked=0;
    auto check=[&](std::vector<ParticleGpu> particles,const std::vector<float>& supports,bool forceHashFailure=false,GranularGridProfile profile=GranularGridProfile::Legacy) {
        const bool basin=profile==GranularGridProfile::Basin;
        const SpatialGrid grid({0,0,0},0.125f,{basin?-128:-64,0,basin?-128:-64},{basin?256u:128u,128,basin?256u:128u});
        NeighborHandle kernel(loadCompute(assets,gridShader("cs_neighbors_test",profile)));
        const auto count=std::uint32_t(particles.size()),words=(count+31)/32,stride=words+4,total=count*stride;
        require(count>0,"Neighbor fixture must contain a slot");
        GpuSpatialKeys keys(assets,count,profile);GpuRadixSort sort(assets,count);GpuCellRanges ranges(assets,count,profile);
        GpuCellHash hash(assets,count,forceHashFailure?2u:0u,64,profile);
        std::vector<std::array<std::uint32_t,2>> sorted(count);
        std::vector<bool> included(count);bool upstreamFailure=false;
        for(std::uint32_t i=0;i<count;++i) {
            const auto& p=particles[i];sorted[i]={UINT32_MAX,i};
            if(!(p.identity[1]&1)) continue;
            if(p.identity[0]>=materials.size()) {upstreamFailure=true;continue;}
            if(materials[p.identity[0]].identity[1]!=0) continue;
            const auto key=grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]});
            if(!key || p.identity[2]!=i || !p.identity[3]) {upstreamFailure=true;continue;}
            sorted[i][0]=*key;included[i]=true;
        }
        std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a[0]<b[0];});
        NeighborHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        NeighborHandle output(bgfx::createDynamicIndexBuffer(total,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE));
        const auto pixels=(total+3)/4,height=(pixels+255)/256;
        NeighborHandle image(bgfx::createTexture2D(256,std::uint16_t(height),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        NeighborHandle staging(bgfx::createTexture2D(256,std::uint16_t(height),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<std::uint32_t> actual(height*256*4);
        for(float support:supports) {
            keys.submit(0,state.value,materialBuffer.value,std::uint32_t(materials.size()));sort.submit(0,keys.records());
            ranges.submit(0,sort.output(),keys.status());hash.submit(0,ranges.ranges(),ranges.metadata());
            const float parameters[]{float(count),float(hash.tableSlots()),float(hash.probeLimit()),support};
            bgfx::setViewMode(1,bgfx::ViewMode::Sequential);bgfx::setUniform(uniform.value,parameters);
            bgfx::setBuffer(0,state.value,bgfx::Access::Read);bgfx::setBuffer(1,sort.output(),bgfx::Access::Read);
            bgfx::setBuffer(2,ranges.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash.table(),bgfx::Access::Read);
            bgfx::setBuffer(4,hash.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,output.value,bgfx::Access::ReadWrite);
            bgfx::dispatch(1,kernel.value,dispatchGroups(count));
            bgfx::setUniform(uniform.value,parameters);bgfx::setBuffer(0,output.value,bgfx::Access::Read);
            bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,dispatchGroups(pixels));
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;
            bgfx::blit(2,dst,src);const auto ready=bgfx::read(dst,actual.data());
            while(bgfx::frame()<ready) SDL_PumpEvents();
            for(std::uint32_t query=0;query<count;++query) {
                const auto slot=sorted[query][1];const auto& a=particles[slot];
                const auto status=upstreamFailure || forceHashFailure?2u:!std::isfinite(support) || support<0?3u:!included[slot]?1u:0u;
                const auto base=query*stride;std::vector<std::uint32_t> expected(words);
                std::uint32_t accepted=0,candidates=0;
                if(status==0) {
                    const auto cell=*grid.coordinates({a.positionRadius[0],a.positionRadius[1],a.positionRadius[2]});
                    const auto radius=std::int64_t(std::ceil(std::min(double(support)*8,basin?255.0:127.0)));
                    for(std::uint32_t j=0;j<count;++j) {
                        if(!included[j]) continue;
                        const auto& b=particles[j];
                        const auto other=*grid.coordinates({b.positionRadius[0],b.positionRadius[1],b.positionRadius[2]});
                        bool candidate=true;
                        for(unsigned axis=0;axis<3;++axis) if(std::abs(std::int64_t(cell[axis])-other[axis])>radius) candidate=false;
                        if(candidate) ++candidates;
                        // All-pairs membership oracle intentionally does not use
                        // candidate cells or the GPU's hash/range implementation.
                        double distance=0;
                        for(unsigned axis=0;axis<3;++axis) {const double d=double(a.positionRadius[axis])-b.positionRadius[axis];distance+=d*d;}
                        if(j!=slot && distance<=double(support)*support) {expected[j/32]|=1u<<(j%32);++accepted;}
                    }
                }
                for(std::uint32_t w=0;w<words;++w) if(actual[base+w]!=expected[w])
                    throw std::runtime_error("GPU neighbor membership mismatch at slot "+std::to_string(slot)+" support "+std::to_string(support));
                require(actual[base+words]==accepted,"GPU neighbor accepted count mismatch");
                require(actual[base+words+1]==candidates,"GPU neighbor candidate count mismatch");
                require(actual[base+words+3]==status,"GPU neighbor failure/exclusion status mismatch");
                const auto probes=actual[base+words+2];
                require(status==0?probes>0 && probes<=(basin?8388608u:2097152u)*hash.probeLimit():probes==0,"GPU neighbor probe count invalid");
                ++queriesChecked;
            }
            ++cases;
        }
    };
    auto fixture=[](std::uint32_t count,bool dense) {
        std::vector<ParticleGpu> p(count);
        for(std::uint32_t i=0;i<count;++i) {
            p[i].positionRadius[0]=dense?0.f:(int((i*37)%65)-32)/64.f;
            p[i].positionRadius[1]=dense?1.f:0.5f+float((i*19)%65)/64.f;
            p[i].positionRadius[2]=dense?0.f:(int((i*13)%65)-32)/64.f;
            p[i].positionRadius[3]=0.0625f;p[i].identity[0]=i%4==3?3:0;
            p[i].identity[1]=1;p[i].identity[2]=i;p[i].identity[3]=1;
        }
        return p;
    };
    for(std::uint32_t count:{1u,31u,32u,33u,127u,128u,129u,255u,256u,257u}) {
        auto p=fixture(count,false);
        for(std::uint32_t i=0;i<count;++i) {
            if(i%13==1) p[i].identity[0]=1;
            if(i%13==2) p[i].identity[0]=2;
            if(i%13==3) {p[i].identity[1]=0;p[i].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();}
        }
        check(p,{0,0.0625f,0.125f,0.25f,0.375f,0.5f});
    }
    check(fixture(1024,true),{0,0.125f}); // 1,023 neighbors; no per-particle cap.
    auto boundaries=fixture(8,false);
    const std::array<std::array<float,3>,8> positions{{{-8,0,-8},{-7.875f,0,-8},{7.875f,0,-8},{7.9375f,0,-8},
        {-0.125f,1,0},{0,1,0},{0.125f,1,0},{0,1.125f,0}}};
    for(unsigned i=0;i<positions.size();++i) for(unsigned axis=0;axis<3;++axis) boundaries[i].positionRadius[axis]=positions[i][axis];
    check(boundaries,{0,0.125f,0.25f});
    check(fixture(2,false),{32,std::numeric_limits<float>::max()});
    check(fixture(1,false),{-1,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()});
    auto tiny=fixture(4,true);
    tiny[0].positionRadius[0]=0;
    tiny[1].positionRadius[0]=std::numeric_limits<float>::denorm_min();
    tiny[2].positionRadius[0]=-std::numeric_limits<float>::denorm_min();
    tiny[3].positionRadius[0]=1e-20f;
    check(tiny,{0,std::numeric_limits<float>::denorm_min(),1e-20f,-std::numeric_limits<float>::denorm_min()});
    auto failed=fixture(129,false);failed[0].positionRadius[0]=8;check(failed,{0.125f});
    failed=fixture(129,false);failed[0].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(failed,{0.125f});
    check(fixture(129,false),{0.125f},true);
    auto empty=fixture(129,false);for(auto& p:empty) p.identity[1]=0;check(empty,{0.125f});
    // Basin outer half, axis strides, cell seams, and largest valid key.
    auto basin=fixture(16,false);
    const float edge=std::nextafter(16.f,0.f);
    const std::array<std::array<float,3>,16> basinPositions{{
        {-16,0,-16},{-15.875f,0,-16},{edge,edge,edge},{15.875f,edge,edge},
        {-11.5f,6,-7.5f},{-11.375f,6,-7.5f},{11.5f,6,7.5f},{11.5f,6,7.625f},
        {-8.125f,4,0},{-8,4,0},{8,4,0},{8.125f,4,0},
        {0,4,-8.125f},{0,4,-8},{0,4,8},{0,4.125f,8}}};
    for(unsigned i=0;i<basin.size();++i)
        for(unsigned axis=0;axis<3;++axis) basin[i].positionRadius[axis]=basinPositions[i][axis];
    check(basin,{0,0.125f,0.25f},false,GranularGridProfile::Basin);
    auto wide=fixture(2,false);wide[0].positionRadius[0]=wide[0].positionRadius[2]=-16;wide[0].positionRadius[1]=0;
    for(unsigned axis=0;axis<3;++axis) wide[1].positionRadius[axis]=edge;
    check(wide,{48,std::numeric_limits<float>::max()},false,GranularGridProfile::Basin);
    for(unsigned axis=0;axis<3;++axis) {
        auto invalid=basin;invalid[0].positionRadius[axis]=16;
        check(invalid,{0.125f},false,GranularGridProfile::Basin);
    }
    check(basin,{0.125f},true,GranularGridProfile::Basin);
    check(tiny,{0,std::numeric_limits<float>::denorm_min()},false,GranularGridProfile::Basin);
    std::cout<<"GPU support-aware neighbors passed: "<<cases<<" fixtures, "<<queriesChecked
             <<" exact all-pairs membership queries, dense 1023-neighbor cells and failure gates\n";
}
}
