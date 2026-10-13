#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace pw::reference {
using FrictionVector=std::array<double,3>;
inline double frictionDot(const FrictionVector& a,const FrictionVector& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline bool frictionFinite(const FrictionVector& v){return std::isfinite(v[0])&&std::isfinite(v[1])&&std::isfinite(v[2]);}
// Independent double oracle for the tangential position constraint in Macklin
// et al. 2014, section 6.1. Normal support depth comes from the contact solve;
// displacement includes corrections already applied during this substep.
inline FrictionVector positionFriction(const FrictionVector& displacement,const FrictionVector& normal,
    double supportDepth,double staticFriction,double kineticFriction) {
    if(!frictionFinite(displacement) || !frictionFinite(normal) ||
       std::abs(frictionDot(normal,normal)-1)>1e-12 || !std::isfinite(supportDepth) || supportDepth<0 ||
       !std::isfinite(staticFriction) || !std::isfinite(kineticFriction) || kineticFriction<0 ||
       staticFriction<kineticFriction || staticFriction>1)
        throw std::invalid_argument("Invalid position friction constraint");
    FrictionVector tangent{};const double along=frictionDot(displacement,normal);
    for(unsigned axis=0;axis<3;++axis)tangent[axis]=displacement[axis]-along*normal[axis];
    const double length=std::hypot(tangent[0],tangent[1],tangent[2]);
    if(!frictionFinite(tangent) || !std::isfinite(length))throw std::invalid_argument("Position friction arithmetic overflow");
    if(length==0 || supportDepth==0)return {};
    const double fraction=length<=staticFriction*supportDepth?1:std::min(1.0,kineticFriction*supportDepth/length);
    for(auto& value:tangent)value*=-fraction;
    return tangent;
}
// Unspent floor traction after a swept impact has already spent friction on
// incoming tangential velocity. Inputs are effective velocities after force/cap.
inline double floorStaticBudget(double tangentSpeed,double normalSpeed,double dt,double mu){
    if(!std::isfinite(tangentSpeed) || tangentSpeed<0 || !std::isfinite(normalSpeed) || normalSpeed<0 ||
       !std::isfinite(dt) || dt<0 || dt>1 || !std::isfinite(mu) || mu<0 || mu>1)
        throw std::invalid_argument("Invalid floor traction budget");
    return std::max(0.0,mu*normalSpeed-tangentSpeed)*dt;
}
struct PositionFrictionPair {FrictionVector a{},b{};};
inline PositionFrictionPair positionFrictionPair(const FrictionVector& relativeDisplacement,const FrictionVector& normal,
    double supportDepth,double inverseMassA,double inverseMassB,double staticFriction,double kineticFriction) {
    if(!std::isfinite(inverseMassA) || !std::isfinite(inverseMassB) || inverseMassA<0 || inverseMassB<0)
        throw std::invalid_argument("Invalid friction inverse mass");
    const auto relative=positionFriction(relativeDisplacement,normal,supportDepth,staticFriction,kineticFriction);
    const double scale=std::max(inverseMassA,inverseMassB);if(scale==0)return {};
    const double a=inverseMassA/scale,b=inverseMassB/scale,wa=a/(a+b),wb=b/(a+b);
    PositionFrictionPair correction;
    for(unsigned axis=0;axis<3;++axis){correction.a[axis]=relative[axis]*wa;correction.b[axis]=-relative[axis]*wb;}
    return correction;
}
}
