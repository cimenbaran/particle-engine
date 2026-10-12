#include "render/Capabilities.hpp"
#include <iostream>
#include <cstdlib>
#define CHECK(x) do { if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;} } while(false)
int main() {
    bgfx::Caps caps{};
    caps.rendererType=bgfx::RendererType::Metal;
    caps.supported=BGFX_CAPS_COMPUTE|BGFX_CAPS_INDEX32|BGFX_CAPS_IMAGE_RW|BGFX_CAPS_DRAW_INDIRECT;
    caps.limits.maxComputeBindings=6;caps.limits.maxInstanceData=2;caps.limits.maxViews=21;
    caps.formats[bgfx::TextureFormat::RGBA32U]=BGFX_CAPS_FORMAT_TEXTURE_IMAGE_WRITE;
    CHECK(pw::capabilityReport(caps,true)["status"]=="supported");
    auto forced=pw::capabilityReport(caps,false);
    CHECK(forced["can_run"]==true && forced["status"]=="degraded");
    CHECK(forced["draw_path"]=="masked_fixed_capacity");
    caps.supported &= ~BGFX_CAPS_DRAW_INDIRECT;
    CHECK(pw::capabilityReport(caps,true)["status"]=="degraded");
    for(const auto flag:{BGFX_CAPS_COMPUTE,BGFX_CAPS_INDEX32,BGFX_CAPS_IMAGE_RW}) {
        auto missing=caps;missing.supported &= ~flag;
        const auto report=pw::capabilityReport(missing,true);
        CHECK(report["status"]=="unavailable" && report["can_run"]==false);
        bool actionable=false;
        for(const auto& check:report["checks"]) if(check["status"]=="unavailable") actionable=!check["action"].get<std::string>().empty();
        CHECK(actionable);
    }
    for(int field=0;field<4;++field) {
        auto missing=caps;
        if(field==0) --missing.limits.maxComputeBindings;
        if(field==1) --missing.limits.maxInstanceData;
        if(field==2) --missing.limits.maxViews;
        if(field==3) missing.formats[bgfx::TextureFormat::RGBA32U]=0;
        CHECK(pw::capabilityReport(missing,true)["can_run"]==false);
    }
    caps.homogeneousDepth=true;caps.originBottomLeft=true;
    const auto report=pw::capabilityReport(caps,true);
    CHECK(report["conventions"]["ndc_depth"]=="minus_one_to_one");
    CHECK(report["conventions"]["origin_bottom_left"]==true);
    CHECK(report["compute"]["workgroup_threads"]==pw::workgroupSize);
    CHECK(report["evidence"]=="advertised_caps_only");
    std::cout<<"Capability classification passed\n";
}
