#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace pw::reference {
using VelocityVector=std::array<double,3>;
struct ContactVelocityPoint {
    std::uint32_t slot;
    VelocityVector predictedPosition{},position{},incomingVelocity{},velocity{};
    double inverseMass=1,restitution=0.1,friction=0.5;
    bool active=true;
    bool operator==(const ContactVelocityPoint&) const=default;
};
struct ContactVelocityResult {
    std::vector<VelocityVector> velocities;
    std::vector<std::uint32_t> degrees;
    std::uint64_t pairs=0;
};
namespace contact_detail {
inline VelocityVector subtract(const VelocityVector& a,const VelocityVector& b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
inline double dot(const VelocityVector& a,const VelocityVector& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline bool finite(const VelocityVector& a) {return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);}
inline bool pair(const ContactVelocityPoint& a,const ContactVelocityPoint& b,double slop) {
    if(!a.active || !b.active || (a.inverseMass==0 && b.inverseMass==0)) return false;
    const auto initial=subtract(a.predictedPosition,b.predictedPosition),current=subtract(a.position,b.position);
    return dot(initial,initial)<=0.125*0.125 || dot(current,current)<=(0.125+slop)*(0.125+slop);
}
inline VelocityVector normal(const ContactVelocityPoint& a,const ContactVelocityPoint& b) {
    auto delta=subtract(a.position,b.position);auto squared=dot(delta,delta);
    if(squared<=1e-12) {delta=subtract(a.predictedPosition,b.predictedPosition);squared=dot(delta,delta);}
    if(squared>1e-12) {for(auto& x:delta) x/=std::sqrt(squared);return delta;}
    delta={0,0,0};const auto axis=(std::min(a.slot,b.slot)*1664525u+std::max(a.slot,b.slot)*1013904223u)%3u;
    delta[axis]=a.slot<b.slot?-1:1;return delta;
}
}
// Test oracle only. No cells, hash lookup, GPU arithmetic or mutable iteration
// inputs. Active points represent compatible physical solids of radius 1/16 m.
inline ContactVelocityResult contactVelocityResponse(std::span<const ContactVelocityPoint> points,
    double bounceThreshold=0.5,double slop=0.0001,double maxDisplacement=0.25) {
    if(points.size()>UINT32_MAX || !std::isfinite(bounceThreshold) || bounceThreshold<0 ||
       !std::isfinite(slop) || slop<0 || slop>0.0625 || !std::isfinite(maxDisplacement) || maxDisplacement<0 || maxDisplacement>32)
        throw std::invalid_argument("Invalid contact velocity profile");
    const double bound=maxDisplacement+1e-5;
    std::unordered_set<std::uint32_t> slots;
    for(const auto& p:points) {
        if(p.slot==UINT32_MAX || !slots.insert(p.slot).second) throw std::invalid_argument("Invalid contact velocity slot");
        if(!p.active) continue;
        const auto displacement=contact_detail::subtract(p.position,p.predictedPosition);
        if(!contact_detail::finite(p.position) || !contact_detail::finite(p.predictedPosition) ||
           !contact_detail::finite(p.incomingVelocity) || !contact_detail::finite(p.velocity) ||
           !std::isfinite(p.inverseMass) || p.inverseMass<0 || p.inverseMass>std::numeric_limits<float>::max() ||
           (p.inverseMass>0 && p.inverseMass<std::numeric_limits<float>::min()) || !std::isfinite(p.restitution) || p.restitution<0 || p.restitution>1 ||
           !std::isfinite(p.friction) || p.friction<0 || p.friction>1 || contact_detail::dot(displacement,displacement)>bound*bound)
            throw std::invalid_argument("Invalid contact velocity snapshot/material/displacement");
    }
    ContactVelocityResult result;result.degrees.resize(points.size());result.velocities.reserve(points.size());
    for(const auto& p:points) result.velocities.push_back(p.velocity);
    for(std::size_t i=0;i<points.size();++i) for(std::size_t j=i+1;j<points.size();++j)
        if(contact_detail::pair(points[i],points[j],slop)) {++result.degrees[i];++result.degrees[j];++result.pairs;}
    for(std::size_t i=0;i<points.size();++i) for(std::size_t j=i+1;j<points.size();++j) {
        const auto& a=points[i];const auto& b=points[j];if(!contact_detail::pair(a,b,slop)) continue;
        const auto normal=contact_detail::normal(a,b),relative=contact_detail::subtract(a.velocity,b.velocity);
        const auto incoming=contact_detail::subtract(a.incomingVelocity,b.incomingVelocity);
        const double currentNormal=contact_detail::dot(relative,normal),incomingNormal=contact_detail::dot(incoming,normal);
        double normalChange=0,support=0;
        if(incomingNormal<=0) {
            const double target=incomingNormal< -bounceThreshold?-std::min(a.restitution,b.restitution)*incomingNormal:0;
            normalChange=target-currentNormal;
            support=std::max(std::abs(currentNormal-incomingNormal),target-incomingNormal);
        } else if(currentNormal<0) {
            // Coupled contacts can introduce approach absent from the original
            // impact snapshot. Resolve it without inventing restitution energy.
            normalChange=-currentNormal;support=normalChange;
        }
        VelocityVector tangent{};for(unsigned axis=0;axis<3;++axis) tangent[axis]=relative[axis]-currentNormal*normal[axis];
        const double speed=std::sqrt(contact_detail::dot(tangent,tangent));
        const double friction=std::min(speed,std::sqrt(a.friction*b.friction)*support);
        VelocityVector change{};for(unsigned axis=0;axis<3;++axis)
            change[axis]=normal[axis]*normalChange-(speed>0?tangent[axis]*(friction/speed):0);
        const auto degree=std::max(a.inverseMass>0?result.degrees[i]:0u,b.inverseMass>0?result.degrees[j]:0u);
        const double scale=std::max(a.inverseMass,b.inverseMass),wa=a.inverseMass/scale,wb=b.inverseMass/scale;
        const double factorA=(wa/(wa+wb))/degree,factorB=(wb/(wa+wb))/degree;
        for(unsigned axis=0;axis<3;++axis) {
            result.velocities[i][axis]+=change[axis]*factorA;
            result.velocities[j][axis]-=change[axis]*factorB;
        }
    }
    for(std::size_t i=0;i<points.size();++i) if(points[i].active && !contact_detail::finite(result.velocities[i]))
        throw std::invalid_argument("Non-finite contact velocity result");
    return result;
}
}
