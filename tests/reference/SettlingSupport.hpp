#pragma once
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace pw::reference {
struct SettlingBody {
    std::array<double,3> position{};
    bool alive=true,solid=true,dynamic=true;
};
struct SettlingSupport {bool enter=false,hold=false;};
// Independent all-pairs geometric oracle for the initial common-radius profile.
// Candidate support is local evidence, not a proof of a ground-connected stack.
inline SettlingSupport settlingSupport(std::span<const SettlingBody> bodies,std::size_t slot,std::array<double,3> sphereCentre={0,2,0},double sphereRadius=0) {
    if(slot>=bodies.size()) throw std::out_of_range("Settling support slot");
    const auto& body=bodies[slot];SettlingSupport result;
    if(!body.alive || !body.solid || !body.dynamic) return result;
    for(double x:body.position)if(!std::isfinite(x))return result;
    auto contact=[&](double gap){
        // Deeply intersecting bodies must resolve before they can qualify.
        if(gap < -0.00625)return;
        result.enter|=gap<=0.0001;result.hold|=gap<=0.002;
    };
    contact(body.position[1]-0.0625);
    if(sphereRadius>0) {
        std::array<double,3> delta{};
        for(unsigned a=0;a<3;++a)delta[a]=body.position[a]-sphereCentre[a];
        const double distance=std::hypot(delta[0],delta[1],delta[2]);
        if(distance>1e-6 && delta[1]>=0.5*distance)contact(distance-sphereRadius-0.0625);
    }
    for(std::size_t other=0;other<bodies.size();++other) {
        const auto& b=bodies[other];if(other==slot || !b.alive || !b.solid)continue;
        std::array<double,3> delta{};double squared=0;
        for(unsigned axis=0;axis<3;++axis){delta[axis]=body.position[axis]-b.position[axis];squared+=delta[axis]*delta[axis];}
        const double distance=std::sqrt(squared);
        // No support from coincident centres, horizontal contacts or bodies above.
        if(!std::isfinite(distance) || distance<=1e-6 || delta[1]<0.5*distance)continue;
        contact(distance-0.125);
    }
    return result;
}
}
