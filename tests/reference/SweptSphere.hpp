#pragma once
#include "SweptBounds.hpp"
namespace pw::reference {
struct SphereCollider {SweepVector centre{0,2,0};double radius=1;};
struct SphereSweepResult {
    SweepState state;
    bool hit=false;
    double time=0;
    SweepVector normal{};
};
// Continuous linear sphere versus one fixed sphere. Initial overlap is an error;
// overlap recovery and choosing among multiple colliders are separate policies.
inline SphereSweepResult sweepSphere(SweepState input,double dt,double particleRadius,
    const SphereCollider& collider,double restitution,double friction,double bounceThreshold=0.5) {
    const double bound=std::numeric_limits<float>::max();
    if(!std::isfinite(dt) || dt<0 || dt>1 || !std::isfinite(particleRadius) || particleRadius<=0 || particleRadius>bound ||
       !std::isfinite(collider.radius) || collider.radius<=0 || collider.radius>bound ||
       !std::isfinite(restitution) || restitution<0 || restitution>1 || !std::isfinite(friction) || friction<0 || friction>1 ||
       !std::isfinite(bounceThreshold) || bounceThreshold<0 || bounceThreshold>bound)
        throw std::invalid_argument("Invalid sphere sweep profile");
    SweepVector offset{};
    for(unsigned a=0;a<3;++a){
        for(double v:{input.position[a],input.velocity[a],collider.centre[a]})
            if(!std::isfinite(v) || std::abs(v)>bound)throw std::invalid_argument("Invalid sphere sweep state");
        offset[a]=input.position[a]-collider.centre[a];
    }
    const double radius=particleRadius+collider.radius;
    const double distance=std::hypot(offset[0],offset[1],offset[2]);
    if(distance<radius)throw std::invalid_argument("Sphere sweep starts inside collider");
    SphereSweepResult result{input};
    auto advance=[&](double time){for(unsigned a=0;a<3;++a)result.state.position[a]+=result.state.velocity[a]*time;};
    if(dt==0)return result;
    const double speed=std::hypot(input.velocity[0],input.velocity[1],input.velocity[2]);
    if(speed==0){advance(dt);return result;}
    SweepVector direction{},perpendicular{};double approach=0;
    for(unsigned axis=0;axis<3;++axis){direction[axis]=input.velocity[axis]/speed;approach-=offset[axis]*direction[axis];}
    if(approach<=0){advance(dt);return result;}
    for(unsigned axis=0;axis<3;++axis)perpendicular[axis]=std::fma(direction[axis],approach,offset[axis]);
    const double missDistance=std::hypot(perpendicular[0],perpendicular[1],perpendicular[2]);
    if(missDistance>=radius){advance(dt);return result;} // tangent touch has no incoming normal speed
    const double chord=std::sqrt((radius-missDistance)*(radius+missDistance));
    // Stable near-surface entry root, with a geometric discriminant that avoids
    // subtracting nearly equal squared flight distances on a distant head-on ray.
    const double c=(distance-radius)*(distance+radius);
    const double time=c/(speed*(approach+chord));
    if(time>dt){advance(dt);return result;}
    // Form impact geometry directly: subtracting two large world positions at
    // impact could erase the collider radius even in double precision.
    SweepVector normal{};
    for(unsigned axis=0;axis<3;++axis)normal[axis]=perpendicular[axis]-direction[axis]*chord;
    const double length=std::hypot(normal[0],normal[1],normal[2]);
    if(length==0 || !std::isfinite(length))throw std::runtime_error("Invalid sphere impact normal");
    for(double& v:normal)v/=length;
    double incoming=0;for(unsigned axis=0;axis<3;++axis)incoming+=input.velocity[axis]*normal[axis];
    if(incoming>=0){result.state=input;advance(dt);return result;}
    const double outgoing=-incoming>bounceThreshold?-restitution*incoming:0;
    const double impulse=outgoing-incoming;
    SweepVector tangent{};for(unsigned axis=0;axis<3;++axis)tangent[axis]=input.velocity[axis]-normal[axis]*incoming;
    const double tangentSpeed=std::hypot(tangent[0],tangent[1],tangent[2]);
    const double scale=tangentSpeed>0?std::max(0.0,1-friction*impulse/tangentSpeed):0;
    for(unsigned axis=0;axis<3;++axis){
        result.state.position[axis]=collider.centre[axis]+normal[axis]*radius;
        result.state.velocity[axis]=normal[axis]*outgoing+tangent[axis]*scale;
    }
    result.hit=true;result.time=time;result.normal=normal;advance(dt-time);
    return result;
}
}
