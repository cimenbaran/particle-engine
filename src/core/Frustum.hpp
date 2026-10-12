#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace pw {
using Frustum=std::array<std::array<float,4>,6>;
// Column-major view-projection matrix, inward normalized planes.
inline Frustum extractFrustum(const float* matrix,bool homogeneousDepth) {
    Frustum result{};
    for(unsigned plane=0;plane<6;++plane) {
        const unsigned axis=plane/2;
        const float sign=plane%2==0?1.0f:-1.0f;
        for(unsigned column=0;column<4;++column)
            result[plane][column]=(plane==4 && !homogeneousDepth?0.0f:matrix[column*4+3])+sign*matrix[column*4+axis];
        const auto& p=result[plane];
        const float length=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if(!std::isfinite(length) || length<=0) throw std::runtime_error("Invalid camera frustum");
        for(auto& value:result[plane]) {value/=length;if(!std::isfinite(value)) throw std::runtime_error("Invalid camera frustum");}
    }
    return result;
}
}
