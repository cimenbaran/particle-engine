#include "GpuPrefixScan.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
namespace pw {
namespace {
template<class T> struct ScanHandle {
    T value;
    explicit ScanHandle(T handle):value(handle) {if(!bgfx::isValid(value)) throw std::runtime_error("Scan test resource unavailable");}
    ~ScanHandle() {bgfx::destroy(value);}
    ScanHandle(const ScanHandle&)=delete;
    ScanHandle& operator=(const ScanHandle&)=delete;
};
}
void runGpuPrefixScanTest(const std::filesystem::path& assets) {
    ScanHandle exporter(loadCompute(assets,"cs_scan_export"));
    ScanHandle uniform(bgfx::createUniform("u_scan",bgfx::UniformType::Vec4));
    bool rejected=false;
    try {GpuPrefixScan invalid(assets,1048577);} catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected) throw std::runtime_error("Oversized scan was accepted");
    for(auto arithmetic:{GpuPrefixScan::Arithmetic::Wrapping,GpuPrefixScan::Arithmetic::Saturating})
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u,16383u,16384u,16385u,65535u,65536u,65537u,1048576u}) {
        GpuPrefixScan scan(assets,count,arithmetic);
        const auto add=[&](std::uint32_t a,std::uint32_t b){return arithmetic==GpuPrefixScan::Arithmetic::Saturating?std::uint32_t(std::min(std::uint64_t(a)+b,std::uint64_t(UINT32_MAX))):a+b;};
        rejected=false;
        try {scan.submit(0,scan.output());} catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected) throw std::runtime_error("Aliased scan input was accepted");
        for(unsigned pattern=0;pattern<4;++pattern) {
            std::vector<std::uint32_t> input(std::max(1u,count));
            for(std::uint32_t i=0;i<count;++i)
                input[i]=pattern==0?0u:pattern==1?1u:pattern==2?(i*1664525u+1013904223u)%2u:(i*1664525u+1013904223u);
            ScanHandle upload(bgfx::createDynamicIndexBuffer(bgfx::copy(input.data(),std::uint32_t(input.size()*4)),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
            const auto height=std::uint16_t((std::max(1u,count)+255u)/256u);
            ScanHandle image(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
            ScanHandle staging(bgfx::createTexture2D(256,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
            std::vector<std::array<std::uint32_t,4>> actual(std::size_t(height)*256);
            scan.submit(0,upload.value);
            const float parameters[]{float(count),0,0,0};bgfx::setUniform(uniform.value,parameters);
            bgfx::setBuffer(0,scan.output(),bgfx::Access::Read);bgfx::setBuffer(1,scan.total(),bgfx::Access::Read);
            bgfx::setBuffer(2,upload.value,bgfx::Access::Read);bgfx::setImage(3,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exporter.value,dispatchGroups(std::max(1u,count)));
            bgfx::TextureRegion source;source.handle=image.value;bgfx::TextureRegion destination;destination.handle=staging.value;
            bgfx::blit(2,destination,source);const auto ready=bgfx::read(destination,actual.data());
            while(bgfx::frame()<ready) SDL_PumpEvents();
            std::uint32_t total=0;for(std::uint32_t i=0;i<count;++i) total=add(total,input[i]);
            std::uint32_t prefix=0;
            for(std::uint32_t i=0;i<std::max(1u,count);++i) {
                const std::array<std::uint32_t,4> expected{prefix,total,input[i],i};
                if(actual[i]!=expected) throw std::runtime_error("GPU prefix scan mismatch at count "+std::to_string(count)+
                    " pattern "+std::to_string(pattern)+" index "+std::to_string(i));
                prefix=add(prefix,input[i]);
            }
        }
        std::cout<<"GPU exclusive prefix scan passed: "<<count<<" elements, zero/one/mixed sums; "<<(arithmetic==GpuPrefixScan::Arithmetic::Saturating?"saturating":"wrapping")<<"\n";
    }
}
}
