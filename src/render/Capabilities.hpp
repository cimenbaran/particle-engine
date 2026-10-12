#pragma once
#include <bgfx/bgfx.h>
#include "core/ComputeDispatch.hpp"
#include <nlohmann/json.hpp>
#include <string>
namespace pw {
// Describes advertised support. Device execution is verified by the GPU suite.
inline nlohmann::json capabilityReport(const bgfx::Caps& caps, bool allowIndirect) {
    using nlohmann::json;
    json checks=json::array();
    bool available=true;
    auto require=[&](const char* name,bool passed,json actual,json minimum,const char* action) {
        available &= passed;
        checks.push_back({{"feature",name},{"status",passed?"supported":"unavailable"},
            {"actual",actual},{"required",minimum},{"action",passed?"":action}});
    };
    require("compute",caps.supported&BGFX_CAPS_COMPUTE,bool(caps.supported&BGFX_CAPS_COMPUTE),true,
        "Use a GPU and driver with compute support on Metal or D3D11.");
    require("compute_bindings",caps.limits.maxComputeBindings>=6,caps.limits.maxComputeBindings,6,
        "The pool auditor needs six simultaneous compute bindings.");
    require("instance_data_slots",caps.limits.maxInstanceData>=2,caps.limits.maxInstanceData,2,
        "Particle rendering needs two vec4 instance-data slots.");
    require("index32",caps.supported&BGFX_CAPS_INDEX32,bool(caps.supported&BGFX_CAPS_INDEX32),true,
        "Pool and visibility lists require 32-bit index buffers.");
    require("image_read_write",caps.supported&BGFX_CAPS_IMAGE_RW,bool(caps.supported&BGFX_CAPS_IMAGE_RW),true,
        "GPU startup validation and diagnostics require compute images.");
    const auto format=caps.formats[bgfx::TextureFormat::RGBA32U];
    require("rgba32u_image_write",format&BGFX_CAPS_FORMAT_TEXTURE_IMAGE_WRITE,
        bool(format&BGFX_CAPS_FORMAT_TEXTURE_IMAGE_WRITE),true,
        "GPU validation and counter export require RGBA32U image writes.");
    require("views",caps.limits.maxViews>=21,caps.limits.maxViews,21,
        "The fixed frame graph uses view IDs through 20.");
    const bool indirectAvailable=caps.supported&BGFX_CAPS_DRAW_INDIRECT;
    const bool indirect=allowIndirect && indirectAvailable;
    checks.push_back({{"feature","indirect_draw"},{"status",indirect?"supported":"degraded"},
        {"available",indirectAvailable},{"enabled",indirect},
        {"action",indirect?"":"Use masked fixed-capacity instancing; no synchronous count readback is required."}});
    return {{"schema_version",1},{"status",!available?"unavailable":indirect?"supported":"degraded"},
        {"can_run",available},{"evidence","advertised_caps_only"},
        {"renderer_type",int(caps.rendererType)},{"gpu_vendor_id",caps.vendorId},{"gpu_device_id",caps.deviceId},
        {"checks",checks},{"draw_path",indirect?"gpu_indirect":"masked_fixed_capacity"},
        {"conventions",{{"handedness","right"},{"up_axis","Y"},{"distance_unit","metre"},
            {"ndc_depth",caps.homogeneousDepth?"minus_one_to_one":"zero_to_one"},
            {"origin_bottom_left",caps.originBottomLeft}}},
        {"compute",{{"workgroup_threads",workgroupSize},{"workgroup_limit_probe","not exposed by pinned bgfx API"},
            {"buffer_read_write_probe","verified by startup GPU tests, not a separate capability bit"},
            {"build_variants",json::array({128,256})},{"selection","compile_time"}}},
        {"runtime_validation","Shader creation, buffer/image access, blits and readback still require successful GPU tests."}};
}
}
