#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
namespace pw {
using BrushPoint=std::array<float,3>;
struct BrushRay {BrushPoint origin,direction;};
struct BrushBox {BrushPoint minimum,maximum;};
inline std::optional<BrushRay> pointerRay(const BrushPoint& eye,const BrushPoint& target,
                                        float u,float v,float aspect,float fovDegrees=55) {
    if(!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(aspect) ||
       !std::isfinite(fovDegrees) || u<0 || u>1 || v<0 || v>1 || aspect<=0 || fovDegrees<=0 || fovDegrees>=179) return {};
    auto normalize=[](BrushPoint& point) {
        const float length=std::sqrt(point[0]*point[0]+point[1]*point[1]+point[2]*point[2]);
        if(!std::isfinite(length) || length<1e-6f) return false;
        for(auto& value:point) value/=length;
        return true;
    };
    BrushPoint forward{};
    for(unsigned i=0;i<3;++i) {if(!std::isfinite(eye[i]) || !std::isfinite(target[i])) return {};forward[i]=target[i]-eye[i];}
    if(!normalize(forward)) return {};
    BrushPoint right{-forward[2],0,forward[0]};if(!normalize(right)) return {};
    BrushPoint up{-right[2]*forward[1],right[2]*forward[0]-right[0]*forward[2],right[0]*forward[1]};
    const float extent=std::tan(fovDegrees*.00872664626f);
    BrushPoint direction{};
    for(unsigned i=0;i<3;++i) direction[i]=forward[i]+right[i]*(2*u-1)*extent*aspect+up[i]*(1-2*v)*extent;
    if(!normalize(direction)) return {};
    return BrushRay{eye,direction};
}
inline std::optional<BrushBox> planeBrush(const BrushRay& ray,float height,float halfSize) {
    if(!std::isfinite(height) || !std::isfinite(halfSize) || height<0 || height>12 || halfSize<.1f || halfSize>3) return {};
    for(unsigned i=0;i<3;++i) if(!std::isfinite(ray.origin[i]) || !std::isfinite(ray.direction[i])) return {};
    if(std::abs(ray.direction[1])<1e-6f) return {};
    const float distance=(height-ray.origin[1])/ray.direction[1];
    if(!std::isfinite(distance) || distance<=.1f || distance>150) return {};
    BrushPoint center{};for(unsigned i=0;i<3;++i) center[i]=ray.origin[i]+ray.direction[i]*distance;
    if(std::abs(center[0])>8 || std::abs(center[2])>8) return {};
    BrushBox bounds;
    for(unsigned i=0;i<3;++i) {
        const float low=i==1?0.0f:-7.95f,high=i==1?15.0f:7.95f;
        bounds.minimum[i]=std::clamp(center[i]-halfSize,low,high);
        bounds.maximum[i]=std::clamp(center[i]+halfSize,low,high);
    }
    return bounds;
}
// Column-major view-projection, top-left logical window coordinates.
inline std::optional<std::array<float,2>> projectBrushPoint(const BrushPoint& point,const float* matrix,float width,float height) {
    const float w=matrix[3]*point[0]+matrix[7]*point[1]+matrix[11]*point[2]+matrix[15];
    if(!std::isfinite(w) || w<=.1f) return {};
    const float x=(matrix[0]*point[0]+matrix[4]*point[1]+matrix[8]*point[2]+matrix[12])/w;
    const float y=(matrix[1]*point[0]+matrix[5]*point[1]+matrix[9]*point[2]+matrix[13])/w;
    if(!std::isfinite(x) || !std::isfinite(y)) return {};
    return std::array<float,2>{(x+1)*.5f*width,(1-y)*.5f*height};
}
}
