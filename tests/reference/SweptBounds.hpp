#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace pw::reference {
using SweepVector=std::array<double,3>;
struct SweepState {SweepVector position{},velocity{};bool operator==(const SweepState&) const=default;};
struct SweepBounds {SweepVector minimum{-8,0,-8},maximum{8,16,8};double radius=0.0625;};
struct SweepResult {SweepState state;std::uint32_t events=0,planes=0;};
// Independent double-precision time-of-impact oracle for a sphere inside six
// axis-aligned planes. No gravity: callers apply forces before the linear sweep.
inline SweepResult sweepBounds(SweepState input,double dt,double restitution,double friction,
                              double bounceThreshold=0.5,const SweepBounds& bounds={},std::uint32_t budget=16) {
    if(!std::isfinite(dt) || dt<0 || dt>1 || !std::isfinite(restitution) || restitution<0 || restitution>1 ||
       !std::isfinite(friction) || friction<0 || friction>1 || !std::isfinite(bounceThreshold) || bounceThreshold<0 ||
       !std::isfinite(bounds.radius) || bounds.radius<=0 || budget==0 || budget>1024)
        throw std::invalid_argument("Invalid sweep profile");
    SweepVector low{},high{};
    for(unsigned a=0;a<3;++a) {
        low[a]=bounds.minimum[a]+bounds.radius;high[a]=bounds.maximum[a]-bounds.radius;
        if(!std::isfinite(low[a]) || !std::isfinite(high[a]) || low[a]>=high[a] ||
           !std::isfinite(input.position[a]) || !std::isfinite(input.velocity[a]) ||
           std::abs(low[a])>std::numeric_limits<float>::max() || std::abs(high[a])>std::numeric_limits<float>::max() ||
           std::abs(input.velocity[a])>std::numeric_limits<float>::max() ||
           input.position[a]<low[a] || input.position[a]>high[a])
            throw std::invalid_argument("Sweep requires finite state inside inset bounds");
    }
    SweepResult result{input};double remaining=dt;
    while(remaining>0) {
        SweepVector times{};double first=std::numeric_limits<double>::infinity();
        for(unsigned a=0;a<3;++a) {
            const double v=result.state.velocity[a];
            times[a]=v==0?std::numeric_limits<double>::infinity():
                ((v>0?high[a]:low[a])-result.state.position[a])/v;
            first=std::min(first,times[a]);
        }
        const double travel=std::min(first,remaining);
        for(unsigned a=0;a<3;++a) result.state.position[a]+=result.state.velocity[a]*travel;
        if(first>remaining) break;
        if(result.events==budget) throw std::runtime_error("Swept boundary event budget exhausted");
        ++result.events;remaining-=first;
        std::array<bool,3> hit{};double impulseSquared=0,tangentSquared=0;
        // Group genuinely simultaneous contacts; tolerance is roundoff scale,
        // not a physical contact skin that would merge nearby distinct impacts.
        const double tolerance=16*std::numeric_limits<double>::epsilon()*std::max(dt,std::abs(first));
        for(unsigned a=0;a<3;++a) {
            hit[a]=std::isfinite(times[a]) && std::abs(times[a]-first)<=tolerance;
            double& v=result.state.velocity[a];
            if(hit[a]) {
                ++result.planes;result.state.position[a]=v>0?high[a]:low[a];
                const double after=std::abs(v)>bounceThreshold?-restitution*v:0;
                const double impulse=after-v;impulseSquared+=impulse*impulse;v=after;
            } else tangentSquared+=v*v;
        }
        // Coulomb budget in the common tangent space. Simultaneous corner
        // normals are resolved together, avoiding axis-order friction bias.
        const double tangent=std::sqrt(tangentSquared),support=std::sqrt(impulseSquared);
        if(tangent>0) {
            const double scale=std::max(0.0,1-friction*support/tangent);
            for(unsigned a=0;a<3;++a) if(!hit[a]) result.state.velocity[a]*=scale;
        }
        for(unsigned a=0;a<3;++a)
            if(!std::isfinite(result.state.position[a]) || !std::isfinite(result.state.velocity[a]))
                throw std::runtime_error("Nonfinite swept boundary result");
    }
    return result;
}
}
