#include "GpuRadixSort.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
namespace pw {
namespace {
template<class T> struct SortHandle {
    T value;
    explicit SortHandle(T handle):value(handle) {if(!bgfx::isValid(value)) throw std::runtime_error("Sort test resource unavailable");}
    ~SortHandle() {bgfx::destroy(value);}
    SortHandle(const SortHandle&)=delete;
    SortHandle& operator=(const SortHandle&)=delete;
};
}
void runGpuRadixSortTest(const std::filesystem::path& assets) {
    SortHandle exporter(loadCompute(assets,"cs_radix_export"));
    SortHandle uniform(bgfx::createUniform("u_radix",bgfx::UniformType::Vec4));
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u,16385u,65537u,1048576u}) {
        GpuRadixSort sort(assets,count);
        bool rejected=false;
        try {sort.submit(0,sort.output());} catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected) throw std::runtime_error("GPU sort accepted aliased input");
        for(unsigned pattern=count>65536?2u:0u;pattern<3;++pattern) {
            std::vector<std::array<std::uint32_t,2>> input(std::max(1u,count));
            for(std::uint32_t i=0;i<count;++i) {
                auto key=pattern==0?0x80000000u:pattern==1?UINT32_MAX-i:((i*1664525u+1013904223u)&0x800003ffu);
                if(pattern==2 && i%257==0) key=UINT32_MAX;
                if(pattern==2 && i%263==0) key=0;
                input[i]={key,0xa5a50000u^i};
            }
            auto expected=input;
            std::stable_sort(expected.begin(),expected.end(),[](const auto& a,const auto& b) {return a[0]<b[0];});
            SortHandle upload(bgfx::createDynamicIndexBuffer(bgfx::copy(input.data(),std::uint32_t(input.size()*8)),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
            const auto height=std::uint16_t((std::max(1u,count)+255u)/256u);
            SortHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
            SortHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
            std::vector<std::array<std::uint32_t,4>> actual(std::size_t(height)*256);
            sort.submit(0,upload.value);
            const float parameters[]{float(count),0,0,0};bgfx::setUniform(uniform.value,parameters);
            bgfx::setBuffer(0,sort.output(),bgfx::Access::Read);bgfx::setBuffer(1,upload.value,bgfx::Access::Read);
            bgfx::setImage(2,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,dispatchGroups(std::max(1u,count)));
            bgfx::TextureRegion source;source.handle=image.value;bgfx::TextureRegion destination;destination.handle=staging.value;
            bgfx::blit(2,destination,source);const auto ready=bgfx::read(destination,actual.data());
            while(bgfx::frame()<ready) SDL_PumpEvents();
            for(std::uint32_t i=0;i<std::max(1u,count);++i) {
                const std::array<std::uint32_t,4> wanted{expected[i][0],expected[i][1],input[i][0],input[i][1]};
                if(actual[i]!=wanted) throw std::runtime_error("GPU stable radix mismatch at count "+std::to_string(count)+
                    " pattern "+std::to_string(pattern)+" index "+std::to_string(i));
            }
        }
        std::cout<<"GPU stable uint32 key/slot radix sort passed: "<<count<<" records\n";
    }
}
}
