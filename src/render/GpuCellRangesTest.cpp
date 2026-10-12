#include "GpuCellRanges.hpp"
#include "GpuCellOccupancy.hpp"
#include "GpuRadixSort.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct RangeHandle {
    T value;
    explicit RangeHandle(T handle):value(handle) {if(!bgfx::isValid(value)) throw std::runtime_error("Range test resource unavailable");}
    ~RangeHandle() {bgfx::destroy(value);}
    RangeHandle(const RangeHandle&)=delete;
    RangeHandle& operator=(const RangeHandle&)=delete;
};
using Record=std::array<std::uint32_t,2>;
static_assert(sizeof(Record)==8);
}
void runGpuCellRangesTest(const std::filesystem::path& assets) {
    RangeHandle exporter(loadCompute(assets,"cs_range_export")),metaExporter(loadCompute(assets,"cs_pool_stats"));
    RangeHandle uniform(bgfx::createUniform("u_ranges",bgfx::UniformType::Vec4));
    const std::array<std::uint32_t,4> zero{};
    RangeHandle upstream(bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
    auto check=[&](GpuCellRanges& ranges,GpuCellOccupancy& occupancy,GpuRadixSort& sorter,std::vector<Record> input,
                  std::array<std::uint32_t,4> sourceStatus,std::uint32_t errors,bool sortInput) {
        const auto count=std::uint32_t(input.size());auto sorted=input;
        if(sortInput) std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a[0]<b[0];});
        std::vector<std::array<std::uint32_t,3>> expected;
        std::uint32_t valid=0;
        for(std::uint32_t i=0;i<count;) {
            if(sorted[i][0]!=UINT32_MAX) ++valid;
            if(sorted[i][0]>=2097152) {++i;continue;}
            auto end=i+1;while(end<count && sorted[end][0]==sorted[i][0]) {++valid;++end;}
            expected.push_back({sorted[i][0],i,end});i=end;
        }
        if(input.empty()) input.push_back({0,0});
        RangeHandle upload(bgfx::createDynamicIndexBuffer(bgfx::copy(input.data(),std::uint32_t(input.size()*8)),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        RangeHandle source(bgfx::createDynamicIndexBuffer(bgfx::copy(sourceStatus.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        const auto height=std::uint16_t((std::max(1u,count)+255u)/256u);
        RangeHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        RangeHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        RangeHandle metaImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        RangeHandle metaStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<std::array<std::uint32_t,4>> actual(std::size_t(height)*256);std::array<std::uint32_t,4> metadata{};
        auto records=upload.value;
        if(sortInput) {sorter.submit(0,records);records=sorter.output();}
        ranges.submit(0,records,source.value);
        occupancy.submit(0,ranges.ranges(),ranges.metadata(),upstream.value);
        const float parameters[]{float(count),0,0,0};bgfx::setUniform(uniform.value,parameters);
        bgfx::setBuffer(0,ranges.ranges(),bgfx::Access::Read);bgfx::setBuffer(1,ranges.metadata(),bgfx::Access::Read);
        bgfx::setImage(2,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exporter.value,dispatchGroups(std::max(1u,count)));
        bgfx::setBuffer(0,ranges.metadata(),bgfx::Access::Read);bgfx::setImage(1,metaImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,metaExporter.value,1);
        bgfx::TextureRegion src;src.handle=image.value;bgfx::TextureRegion dst;dst.handle=staging.value;
        bgfx::blit(2,dst,src);const auto ready=bgfx::read(dst,actual.data());
        src.handle=metaImage.value;dst.handle=metaStaging.value;
        bgfx::blit(2,dst,src);const auto metaReady=bgfx::read(dst,metadata.data());
        while(bgfx::frame()<std::max(ready,metaReady)) SDL_PumpEvents();
        // Read the reduction after the existing range images have drained.
        bgfx::setBuffer(0,occupancy.status(),bgfx::Access::Read);bgfx::setImage(1,metaImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,metaExporter.value,1);
        src.handle=metaImage.value;dst.handle=metaStaging.value;bgfx::blit(2,dst,src);
        std::array<std::uint32_t,4> occupancyStats{};auto occupancyReady=bgfx::read(dst,occupancyStats.data());while(bgfx::frame()<occupancyReady)SDL_PumpEvents();
        std::uint32_t maximum=0;for(const auto& range:expected)maximum=std::max(maximum,range[2]-range[1]);
        const std::array<std::uint32_t,4> wantedOccupancy{errors?1u:0u,errors?0u:std::uint32_t(expected.size()),errors?0u:valid,errors?0u:maximum};
        if(occupancyStats!=wantedOccupancy)throw std::runtime_error("GPU cell occupancy mismatch at capacity "+std::to_string(count));
        const auto occupied=std::uint32_t(expected.size());
        const std::array<std::uint32_t,4> expectedMeta{errors?0u:occupied,errors?0u:valid,errors,occupied};
        if(metadata!=expectedMeta) throw std::runtime_error("GPU range metadata mismatch, count "+std::to_string(count)+
            " expected error "+std::to_string(errors)+" actual "+std::to_string(metadata[2]));
        for(std::uint32_t i=0;i<std::max(1u,count);++i) {
            std::array<std::uint32_t,4> wanted{UINT32_MAX,UINT32_MAX,UINT32_MAX,i};
            if(i<occupied) wanted={expected[i][0],expected[i][1],expected[i][2],i};
            if(actual[i]!=wanted) throw std::runtime_error("GPU range interval mismatch at count "+std::to_string(count)+" range "+std::to_string(i));
        }
    };
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u,16385u,65537u,1048576u}) {
        GpuCellRanges ranges(assets,count);GpuCellOccupancy occupancy(assets,count);GpuRadixSort sorter(assets,count);
        const unsigned first=count>16385?1:0,last=count>16385?2:3;
        for(unsigned pattern=first;pattern<last;++pattern) {
            std::vector<Record> input(count);std::array<std::uint32_t,4> source{};
            for(std::uint32_t i=0;i<count;++i) {
                const auto key=pattern==0?17u:pattern==2 && i%3==0?UINT32_MAX:(i*1664525u+1013904223u)%2097152u;
                input[i]={key,i};++source[key==UINT32_MAX?1:0];
            }
            check(ranges,occupancy,sorter,std::move(input),source,0,true);
        }
        std::cout<<"GPU sorted cell ranges and occupancy passed: "<<count<<" records\n";
    }
    GpuCellRanges ranges(assets,2);GpuCellOccupancy occupancy(assets,2);GpuRadixSort sorter(assets,2);
    check(ranges,occupancy,sorter,{{2,0},{1,1}},{2,0,0,0},4,false);
    check(ranges,occupancy,sorter,{{UINT32_MAX,0},{1,1}},{1,1,0,0},4,false);
    check(ranges,occupancy,sorter,{{1,0},{2097152,1}},{2,0,0,0},8,false);
    check(ranges,occupancy,sorter,{{1,0},{UINT32_MAX,1}},{1,0,1,0},1,false);
    check(ranges,occupancy,sorter,{{1,0},{UINT32_MAX,1}},{1,0,0,1},2,false);
    check(ranges,occupancy,sorter,{{1,0},{UINT32_MAX,1}},{2,0,0,0},16,false);
    check(ranges,occupancy,sorter,{{1,0},{2,1}},{1,0,0,0},16,false);
    check(ranges,occupancy,sorter,{{1,0},{2,2}},{2,0,0,0},32|64,false);
    check(ranges,occupancy,sorter,{{1,0},{2,0}},{2,0,0,0},64,false);
    check(ranges,occupancy,sorter,{{UINT32_MAX,0},{UINT32_MAX,0}},{0,2,0,0},64,false);
    check(ranges,occupancy,sorter,{{UINT32_MAX,0},{UINT32_MAX,1}},{0,2,0,0},0,false);
    check(ranges,occupancy,sorter,{{1,0},{2,1}},{UINT32_MAX,3,0,0},16,false);
    check(ranges,occupancy,sorter,{{0,0},{2097151,1}},{2,0,0,0},0,false);
    check(ranges,occupancy,sorter,{{1,0},{2,1}},{2,0,0,0},0,false);
    // Inject malformed reduction inputs directly, independent of range generation.
    struct OccupancyCase {std::array<std::uint32_t,6> ranges;std::array<std::uint32_t,4> meta,upstream;std::uint32_t error;};
    const std::array<OccupancyCase,6> invalid{{
        {{{1,0,2,0,0,0}},{{1,2,0,1}},{{1,0,0,0}},2},
        {{{1,0,2,0,0,0}},{{3,2,0,3}},{{0,0,0,0}},1},
        {{{2097152,0,2,0,0,0}},{{1,2,0,1}},{{0,0,0,0}},4},
        {{{1,1,1,0,0,0}},{{1,2,0,1}},{{0,0,0,0}},4},
        {{{1,0,3,0,0,0}},{{1,2,0,1}},{{0,0,0,0}},4},
        {{{1,0,2,0,0,0}},{{0,2,0,0}},{{0,0,0,0}},1}
    }};
    for(const auto& fixture:invalid){
        RangeHandle raw(bgfx::createDynamicIndexBuffer(bgfx::copy(fixture.ranges.data(),24),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        RangeHandle meta(bgfx::createDynamicIndexBuffer(bgfx::copy(fixture.meta.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        RangeHandle source(bgfx::createDynamicIndexBuffer(bgfx::copy(fixture.upstream.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        RangeHandle image(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        RangeHandle staging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        for(bool reset:{false,true}){
            if(reset)occupancy.reset(0,upstream.value);else occupancy.submit(0,raw.value,meta.value,source.value);
            bgfx::setBuffer(0,occupancy.status(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,metaExporter.value,1);
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);
            std::array<std::uint32_t,4> result{};auto ready=bgfx::read(dst,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();
            if(result[0]!=(reset?0u:fixture.error) || (reset && result!=zero))throw std::runtime_error("Occupancy malformed input/reset mismatch");
        }
    }
    std::cout<<"GPU occupancy failure/reset passed: six malformed/upstream inputs and cleared epochs\n";
    std::cout<<"GPU range failure gate passed: upstream errors, exits, order, keys, counts, slot bounds, ownership and recovery\n";
}
}
