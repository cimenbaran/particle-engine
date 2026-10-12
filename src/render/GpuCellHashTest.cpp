#include "GpuCellHash.hpp"
#include "GpuCellRanges.hpp"
#include "GpuRadixSort.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace pw {
namespace {
template<class T> struct HashHandle {
    T value;
    explicit HashHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Hash test allocation failed");}
    ~HashHandle(){bgfx::destroy(value);}
    HashHandle(const HashHandle&)=delete;
};
using Words=std::array<std::uint32_t,4>;
using Range=std::array<std::uint32_t,3>;
std::uint32_t hash(std::uint32_t x) {x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
}
void runGpuCellHashTest(const std::filesystem::path& assets) {
    HashHandle exporter(loadCompute(assets,"cs_hash_export")),stats(loadCompute(assets,"cs_pool_stats"));
    HashHandle uniform(bgfx::createUniform("u_hash",bgfx::UniformType::Vec4));
    auto check=[&](GpuCellHash& index,std::vector<Range> ranges,Words sourceStatus,
                  std::vector<std::uint32_t> queries,Words expectedMeta,std::uint32_t limit,
                  std::uint32_t expectedTotal=UINT32_MAX,std::uint32_t expectedMax=UINT32_MAX,bool throughPipeline=false) {
        std::unordered_map<std::uint32_t,Range> reference;
        for(const auto& r:ranges) reference.emplace(r[0],r);
        if(ranges.empty()) ranges.push_back({UINT32_MAX,0,0});
        const auto flags=BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE;
        HashHandle upload(bgfx::createDynamicIndexBuffer(bgfx::copy(ranges.data(),std::uint32_t(ranges.size()*12)),flags));
        HashHandle source(bgfx::createDynamicIndexBuffer(bgfx::copy(sourceStatus.data(),16),flags));
        HashHandle keys(bgfx::createDynamicIndexBuffer(bgfx::copy(queries.data(),std::uint32_t(queries.size()*4)),flags));
        HashHandle results(bgfx::createDynamicIndexBuffer(std::uint32_t(queries.size()*4),flags));
        const auto height=std::uint16_t((queries.size()+255)/256);
        HashHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        HashHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        HashHandle metaImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        HashHandle metaStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        HashHandle metricImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        HashHandle metricStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::unique_ptr<GpuRadixSort> sorter;
        std::unique_ptr<GpuCellRanges> rangeBuilder;
        std::unique_ptr<HashHandle<bgfx::DynamicIndexBufferHandle>> records,pipelineStatus;
        if(throughPipeline) {
            // Build ranges on the GPU from reversed, duplicate-key records.
            // CPU ranges above are solely the expected-result oracle.
            const auto count=sourceStatus[1];
            std::vector<std::array<std::uint32_t,2>> input(count);
            for(const auto& r:ranges) for(auto i=r[1];i<r[2];++i) input[count-1-i]={r[0],i};
            const Words status{count,0,0,0};
            records=std::make_unique<HashHandle<bgfx::DynamicIndexBufferHandle>>(bgfx::createDynamicIndexBuffer(bgfx::copy(input.data(),count*8),flags));
            pipelineStatus=std::make_unique<HashHandle<bgfx::DynamicIndexBufferHandle>>(bgfx::createDynamicIndexBuffer(bgfx::copy(status.data(),16),flags));
            sorter=std::make_unique<GpuRadixSort>(assets,count);
            rangeBuilder=std::make_unique<GpuCellRanges>(assets,count);
            sorter->submit(0,records->value);rangeBuilder->submit(0,sorter->output(),pipelineStatus->value);
            index.submit(0,rangeBuilder->ranges(),rangeBuilder->metadata());
        } else index.submit(0,upload.value,source.value);
        index.lookup(1,keys.value,results.value,std::uint32_t(queries.size()));
        const float p[]{0,0,0,float(queries.size())};bgfx::setUniform(uniform.value,p);
        bgfx::setBuffer(0,results.value,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exporter.value,dispatchGroups(std::uint32_t(queries.size())));
        bgfx::setBuffer(0,index.metadata(),bgfx::Access::Read);bgfx::setImage(1,metaImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,stats.value,1);
        bgfx::setBuffer(0,index.metrics(),bgfx::Access::Read);bgfx::setImage(1,metricImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,stats.value,1);
        std::vector<Words> actual(std::size_t(height)*256);Words meta{},metrics{};
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;
        bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=metaImage.value;dst.handle=metaStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,meta.data()));
        src.handle=metricImage.value;dst.handle=metricStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,metrics.data()));
        while(bgfx::frame()<ready) SDL_PumpEvents();
        if(meta!=expectedMeta) throw std::runtime_error("GPU hash metadata mismatch: actual "+std::to_string(meta[0])+","+std::to_string(meta[1])+","+std::to_string(meta[2])+","+std::to_string(meta[3]));
        require(std::uint64_t(metrics[0])<=std::uint64_t(sourceStatus[0])*limit && metrics[1]<=limit && metrics[2]==0 && metrics[3]==0,"GPU hash probe metrics invalid");
        if(expectedTotal!=UINT32_MAX) require(metrics[0]==expectedTotal,"GPU hash total probe mismatch");
        if(expectedMax!=UINT32_MAX) require(metrics[1]==expectedMax,"GPU hash maximum probe mismatch");
        for(std::size_t i=0;i<queries.size();++i) {
            const auto it=reference.find(queries[i]);
            const auto status=meta[0]==0?2u:it==reference.end()?0u:1u;
            require(actual[i][2]==status,"GPU hash found/absent/failed status mismatch");
            require(actual[i][0]==(status==1?it->second[1]:0) && actual[i][1]==(status==1?it->second[2]:0),"GPU hash returned incorrect interval");
            require(actual[i][3]<=limit,"GPU hash query exceeded probe bound");
            if(status==2 || queries[i]>=2097152) require(actual[i][3]==0,"GPU hash queried an invalid index/key");
            if(status==1) require(actual[i][3]>0,"GPU hash found key without probing");
        }
    };
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u,16384u,1048576u}) {
        GpuCellHash index(assets,count);
        std::vector<Range> ranges(count);std::vector<std::uint32_t> keys;
        for(std::uint32_t i=0;i<count;++i) {ranges[i]={i*2,i,i+1};keys.push_back(i*2);}
        if(keys.empty()) keys={0,2097151,UINT32_MAX};
        check(index,ranges,{count,count,0,count},keys,{1,count,0,0},64);
        // Reuse the same table for missing-key queries, then clear a populated
        // table to empty and verify that no old key survives the rebuild.
        check(index,ranges,{count,count,0,count},{1,2097151,2097152,UINT32_MAX},{1,count,0,0},64);
        if(count) check(index,{{0,0,count}},{1,count,0,1},{0,1,UINT32_MAX},{1,1,0,0},64,1,1);
        check(index,{}, {0,0,0,0},{0,2,2097150},{1,0,0,0},64,0,0);
        std::cout<<"GPU exact-key cell hash passed: "<<count<<" ranges\n";
    }
    GpuCellHash integrated(assets,257);
    check(integrated,{{0,0,129},{17,129,256},{2097151,256,257}},
          {3,257,0,3},{0,1,17,18,2097151,2097152,UINT32_MAX},{1,3,0,0},64,UINT32_MAX,UINT32_MAX,true);
    std::cout<<"GPU sort/range/hash pipeline passed: 257 records, duplicate cells and boundary keys\n";
    std::vector<std::uint32_t> collision;
    for(std::uint32_t key=0;collision.size()<4;++key) if((hash(key)&7)==0) collision.push_back(key);
    std::vector<Range> ranges{{collision[0],0,1},{collision[1],1,2},{collision[2],2,3}};
    GpuCellHash bounded(assets,3,8,3);
    check(bounded,ranges,{3,3,0,3},collision,{1,3,0,0},3,6,3);
    GpuCellHash exhausted(assets,3,8,1);
    check(exhausted,ranges,{3,3,0,3},collision,{0,1,2,4},1,3,1);
    check(exhausted,{{collision[0],0,3}},{1,3,0,1},collision,{1,1,0,0},1,1,1);
    GpuCellHash full(assets,3,2,2);
    check(full,ranges,{3,3,0,3},collision,{0,2,1,2|4},2,5,2);
    GpuCellHash fault(assets,3,8,8);
    check(fault,{{1,0,1},{1,1,2}},{2,2,0,2},{1,2,UINT32_MAX},{0,1,1,8},8);
    check(fault,{{2097152,0,1}},{1,1,0,1},{1,2097152},{0,0,1,16},8,0,0);
    check(fault,{{1,1,1}},{1,1,0,1},{1,2},{0,0,1,16},8,0,0);
    check(fault,{{1,0,2}},{1,1,0,1},{1,2},{0,0,1,16},8,0,0);
    check(fault,{}, {0,0,64,3},{0,1,UINT32_MAX},{0,0,0,1},8,0,0);
    check(fault,{}, {UINT32_MAX,UINT32_MAX,0,UINT32_MAX},{0,1},{0,0,0,2},8,0,0);
    check(fault,{}, {4,4,0,4},{0,1},{0,0,0,2},8,0,0);
    check(fault,{}, {0,1,0,0},{0,1},{0,0,0,2},8,0,0);
    check(fault,{}, {0,0,0,1},{0,1},{0,0,0,2},8,0,0);
    check(fault,{{0,0,1},{2097151,1,3}},{2,3,0,2},{0,1,2097151},{1,2,0,0},8);
    GpuCellHash half(assets,3,4,8);
    check(half,{{0,0,1},{1,1,2}},{2,2,0,2},{0,1,2},{0,2,0,2},8);
    std::cout<<"GPU cell hash collision, probe exhaustion, full table, duplicates, malformed source, strict load and recovery gates passed\n";
}
}
