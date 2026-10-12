#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace pw {
struct GranularStepProfile {
    float tickSeconds=1.0f/60.0f;
    float speedLimit=12.0f;
    float smallestFeature=0.125f;
    std::uint32_t maxSubsteps=8;
};
struct GranularStepPlan {
    std::uint32_t substeps;
    float substepSeconds;
    float maximumTravel;
};
// A fixed profile gives every GPU particle the same step schedule without a
// synchronous speed readback. The GPU must cap speed after forces and contacts.
inline GranularStepPlan planGranularStep(const GranularStepProfile& p={}) {
    if(!std::isfinite(p.tickSeconds) || p.tickSeconds<=0 || p.tickSeconds>1 ||
       !std::isfinite(p.speedLimit) || p.speedLimit<=0 ||
       !std::isfinite(p.smallestFeature) || p.smallestFeature<=0 ||
       p.maxSubsteps==0 || p.maxSubsteps>64)
        throw std::invalid_argument("Invalid granular substep profile");
    const double halfFeature=double(p.smallestFeature)*0.5;
    // Validate the actual float duration that will be sent to the shader.
    for(std::uint32_t n=1;n<=p.maxSubsteps;++n) {
        const float h=p.tickSeconds/float(n);
        const double travel=double(p.speedLimit)*h;
        if(h>0 && travel<=halfFeature) return {n,h,float(travel)};
    }
    throw std::invalid_argument("Granular speed/feature profile exceeds substep budget");
}
}
