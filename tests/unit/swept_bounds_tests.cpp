#include "core/GranularStep.hpp"
#include "../reference/SweptBounds.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}} while(false)
using namespace pw::reference;
bool near(double a,double b,double e=1e-10) {return std::abs(a-b)<=e;}
double energy(const SweepState& s) {double e=0;for(double v:s.velocity)e+=v*v;return e;}
template<class F> bool rejects(F f) {try{f();return false;}catch(const std::exception&){return true;}}
int main() {
    auto plan=pw::planGranularStep();CHECK(plan.substeps==4 && double(plan.substepSeconds)*12<=0.0625);
    CHECK(near(double(plan.substepSeconds)*plan.substeps,double(1.0f/60),1e-9));
    pw::GranularStepProfile profile;
    for(float speed:{0.01f,1.0f,12.0f,24.0f}) for(float feature:{0.0625f,0.125f,0.5f}) {
        profile.speedLimit=speed;profile.smallestFeature=feature;profile.maxSubsteps=64;plan=pw::planGranularStep(profile);
        CHECK(double(speed)*plan.substepSeconds<=double(feature)*0.5);
        CHECK(near(double(plan.substepSeconds)*plan.substeps,double(profile.tickSeconds),1e-8));
    }
    profile={};profile.speedLimit=100;CHECK(rejects([&]{pw::planGranularStep(profile);}));
    for(float value:{0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        profile={};profile.tickSeconds=value;CHECK(rejects([&]{pw::planGranularStep(profile);}));
        profile={};profile.speedLimit=value;CHECK(rejects([&]{pw::planGranularStep(profile);}));
        profile={};profile.smallestFeature=value;CHECK(rejects([&]{pw::planGranularStep(profile);}));
    }
    profile={};profile.maxSubsteps=0;CHECK(rejects([&]{pw::planGranularStep(profile);}));
    profile={};profile.maxSubsteps=65;CHECK(rejects([&]{pw::planGranularStep(profile);}));
    profile={};profile.tickSeconds=0.0625f;profile.speedLimit=1;profile.smallestFeature=0.125f;
    CHECK(pw::planGranularStep(profile).substeps==1);
    profile.speedLimit=std::nextafter(1.0f,2.0f);CHECK(pw::planGranularStep(profile).substeps==2);
    const SweepState falling{{0,1.0625,0},{0,-10,0}};
    auto r=sweepBounds(falling,0.2,0.5,0);CHECK(r.events==1 && r.planes==1);
    CHECK(near(r.state.position[1],0.5625) && near(r.state.velocity[1],5));CHECK(falling.position[1]==1.0625);
    r=sweepBounds(falling,0.1,0.5,0);CHECK(r.events==1 && near(r.state.position[1],0.0625) && r.state.velocity[1]==5);
    r=sweepBounds({{0,0.0625,0},{0,-0.1,0}},0.1,1,0);CHECK(r.events==1 && r.state.velocity[1]==0 && r.state.position[1]==0.0625);
    r=sweepBounds({{0,0.0625,0},{1,0,0}},0.1,1,1);CHECK(r.events==0 && near(r.state.position[0],0.1));
    r=sweepBounds({{0,0.0625,0},{1,-1,0}},0.1,0,1);CHECK(r.state.velocity==SweepVector({0,0,0}));
    r=sweepBounds({{0,0.0625,0},{1,-1,0}},0.1,0,0.25);CHECK(near(r.state.velocity[0],0.75) && r.state.velocity[1]==0);
    r=sweepBounds({{7.9375,8,7.9375},{2,3,2}},0.1,1,0);CHECK(r.events==1 && r.planes==2);
    CHECK(r.state.velocity==SweepVector({-2,3,-2}));
    // Six faces, high-speed shots, corner impacts and repeated elastic bounces.
    // Independent oracle: unfold the interval and fold the endpoint modulo 2L.
    std::mt19937 random(20261013);std::uniform_real_distribution<double> position(-7.5,7.5),velocity(-1000,1000);
    unsigned cases=0;
    for(unsigned i=0;i<512;++i) {
        SweepState s{{position(random),position(random)+8,position(random)},{velocity(random),velocity(random),velocity(random)}};
        const double dt=0.2,lo[3]{-7.9375,0.0625,-7.9375},width=15.875;
        r=sweepBounds(s,dt,1,0,0,{},128);
        for(unsigned a=0;a<3;++a) {
            double folded=std::fmod(s.position[a]-lo[a]+s.velocity[a]*dt,2*width);if(folded<0)folded+=2*width;
            const double expected=lo[a]+(folded<=width?folded:2*width-folded);
            CHECK(near(r.state.position[a],expected,1e-9));
            CHECK(near(r.state.velocity[a],folded<width?s.velocity[a]:-s.velocity[a],1e-9));
        }
        CHECK(near(energy(r.state),energy(s),1e-8));
        for(double e:{0.0,0.5,1.0}) for(double mu:{0.0,0.5,1.0}) {
            r=sweepBounds(s,dt,e,mu,0.5,{},128);
            CHECK(energy(r.state)<=energy(s)+1e-8);
            for(unsigned a=0;a<3;++a) CHECK(r.state.position[a]>=lo[a]-1e-10 && r.state.position[a]<=lo[a]+width+1e-10);
            SweepState swapped=s;std::swap(swapped.position[0],swapped.position[2]);std::swap(swapped.velocity[0],swapped.velocity[2]);
            auto permuted=sweepBounds(swapped,dt,e,mu,0.5,{},128);
            std::swap(permuted.state.position[0],permuted.state.position[2]);std::swap(permuted.state.velocity[0],permuted.state.velocity[2]);
            for(unsigned a=0;a<3;++a) {
                CHECK(near(r.state.position[a],permuted.state.position[a],1e-9));
                CHECK(near(r.state.velocity[a],permuted.state.velocity[a],1e-9));
            }
            ++cases;
        }
    }
    CHECK(rejects([&]{sweepBounds({{0,8,0},{1000,0,0}},1,1,0,0,{},1);}));
    CHECK(rejects([&]{sweepBounds({{9,8,0},{0,0,0}},0.1,0,0);}));
    CHECK(rejects([&]{sweepBounds(falling,-1,0,0);}));
    CHECK(rejects([&]{sweepBounds(falling,1,1.1,0);}));
    CHECK(rejects([&]{sweepBounds(falling,1,0,-0.1);}));
    CHECK(rejects([&]{sweepBounds(falling,2,0,0);}));
    CHECK(rejects([&]{sweepBounds({{0,8,0},{std::numeric_limits<double>::infinity(),0,0}},0.1,0,0);}));
    const auto zero=sweepBounds(falling,0,1,0);CHECK(zero.state==falling && zero.events==0);
    std::cout<<"Swept bounds: 512 unfolded elastic oracles, "<<cases<<" dissipative cases, analytic contacts and substep profile passed\n";
}
