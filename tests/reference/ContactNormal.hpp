#pragma once
#include "core/AnalyticSphere.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace pw::reference {
struct NormalBody {
    std::array<double,3> position{};
    bool alive=true,solid=true,dynamic=true;
};
enum class NormalSource {None,Plane,Sphere,Particle};
struct ContactNormal {
    std::array<double,3> direction{}; // Toward the queried body, away from collider.
    double gap=std::numeric_limits<double>::infinity();
    NormalSource source=NormalSource::None;
    std::uint32_t feature=0; // Plane x-/x+/y-/y+/z-/z+, or other particle slot.
};
// All-pairs diagnostic oracle, independent of GPU spatial traversal. It describes
// current geometry, not a historical impulse or the sum of simultaneous normals.
inline ContactNormal contactNormal(std::span<const NormalBody> bodies,std::size_t query,AnalyticSphere sphere={}) {
    if(query>=bodies.size())throw std::out_of_range("Contact normal query slot");
    if(bodies.size()>UINT32_MAX)throw std::invalid_argument("Contact normal slot range");
    validateAnalyticSphere(sphere);
    ContactNormal result;
    const auto& body=bodies[query];
    if(!body.alive || !body.solid || !body.dynamic)return result;
    for(const auto& b:bodies)if(b.alive && b.solid)
        for(double v:b.position)if(!std::isfinite(v) || std::abs(v)>16)
            throw std::invalid_argument("Contact normal position outside finite bounded profile");
    auto consider=[&](double gap,std::array<double,3> n,NormalSource source,std::uint32_t feature) {
        // Ascending traversal supplies exact-tie priority: planes, sphere, slots.
        if(gap<=0.002 && gap<result.gap)result={n,gap,source,feature};
    };
    for(unsigned axis=0;axis<3;++axis) {
        const double low=axis==1?0.0625:-7.9375,high=axis==1?15.9375:7.9375;
        std::array<double,3> normal{};normal[axis]=1;
        consider(body.position[axis]-low,normal,NormalSource::Plane,axis*2);
        normal[axis]=-1;
        consider(high-body.position[axis],normal,NormalSource::Plane,axis*2+1);
    }
    if(sphere.radius>0) {
        std::array<double,3> delta{};
        for(unsigned a=0;a<3;++a)delta[a]=body.position[a]-sphere.centre[a];
        const double distance=std::hypot(delta[0],delta[1],delta[2]);
        if(distance==0)throw std::invalid_argument("Sphere centre has no outward normal");
        for(double& v:delta)v/=distance;
        consider(distance-double(sphere.radius)-0.0625,delta,NormalSource::Sphere,0);
    }
    for(std::size_t other=0;other<bodies.size();++other) {
        const auto& b=bodies[other];if(other==query || !b.alive || !b.solid)continue;
        std::array<double,3> normal{};
        for(unsigned a=0;a<3;++a)normal[a]=body.position[a]-b.position[a];
        const double distance=std::hypot(normal[0],normal[1],normal[2]);
        if(distance<=1e-6) {
            // Match the solver's documented antisymmetric coincident-slot policy.
            const auto lo=std::uint32_t(std::min(query,other)),hi=std::uint32_t(std::max(query,other));
            const auto axis=std::uint32_t(lo*1664525u+hi*1013904223u)%3;
            normal={};normal[axis]=query<other?-1:1;
        } else for(double& v:normal)v/=distance;
        consider(distance-0.125,normal,NormalSource::Particle,std::uint32_t(other));
    }
    return result;
}
}
