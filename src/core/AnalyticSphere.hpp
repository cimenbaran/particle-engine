#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace pw {
// Optional fixed interior sphere. Radius zero disables it; particles use radius .0625.
struct AnalyticSphere {std::array<float,3> centre{0,2,0};float radius=0;};
inline void validateAnalyticSphere(const AnalyticSphere& sphere){
    if(!std::isfinite(sphere.radius) || sphere.radius<0)throw std::invalid_argument("Invalid analytic sphere radius");
    for(float value:sphere.centre)if(!std::isfinite(value))throw std::invalid_argument("Invalid analytic sphere centre");
    if(sphere.radius==0)return;
    if(sphere.radius<0.0625f || sphere.radius>4)throw std::invalid_argument("Analytic sphere radius outside supported profile");
    // Keep sphere and plane contact surfaces separated by at least one diameter.
    // They cannot form simultaneous sphere/plane contacts in this profile.
    const double extent=double(sphere.radius)+0.0625+0.125;
    for(unsigned axis=0;axis<3;++axis)
        if(double(sphere.centre[axis])-extent<(axis==1?0.0625:-7.9375) ||
           double(sphere.centre[axis])+extent>(axis==1?15.9375:7.9375))
            throw std::invalid_argument("Analytic sphere must fit inside separated enclosure limits");
}
}
