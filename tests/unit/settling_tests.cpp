#include "core/Settling.hpp"
#include "../reference/SettlingSupport.hpp"
#include <vector>
#include <cstdlib>
#include <iostream>
#include <limits>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
template<class F> bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    using namespace pw;
    SettlingProfile p;
    SettlingObservation o{7,0.05f,true,true,true,true};
    SettlingState state;
    // Exact dwell boundary, then a long steady run with bounded counters.
    for(unsigned tick=1;tick<=29;++tick){state=advanceSettling(state,o);CHECK(!state.candidate && state.quietTicks==tick);}
    state=advanceSettling(state,o);CHECK(state.candidate && state.quietTicks==30);
    for(unsigned tick=0;tick<100000;++tick)state=advanceSettling(state,o);
    CHECK((state==SettlingState{7,30,0,1}));
    // Speed and spatial-contact dead bands retain a candidate but cannot acquire.
    o.speed=0.075f;o.enterContact=false;
    CHECK(advanceSettling(state,o).candidate);
    CHECK(!advanceSettling({},o).candidate && advanceSettling({},o).quietTicks==0);
    o.speed=0.1f;CHECK(advanceSettling(state,o).candidate);
    o.speed=std::nextafter(0.1f,1.0f);CHECK(!advanceSettling(state,o).candidate);
    o.speed=0.05f;o.enterContact=true;
    CHECK(advanceSettling({},o).quietTicks==1);
    o.speed=std::nextafter(0.05f,1.0f);CHECK(advanceSettling({},o).quietTicks==0);
    // Two missing contacts survive; the third releases. Recovery restarts grace.
    o.speed=0;o.enterContact=false;o.holdContact=false;
    auto lost=advanceSettling(state,o);CHECK(lost.candidate && lost.unsupportedTicks==1);
    lost=advanceSettling(lost,o);CHECK(lost.candidate && lost.unsupportedTicks==2);
    CHECK(!advanceSettling(lost,o).candidate);
    o.holdContact=true;lost=advanceSettling(lost,o);CHECK(lost.candidate && lost.unsupportedTicks==0);
    o.holdContact=false;CHECK(advanceSettling(lost,o).unsupportedTicks==1);
    // Recycling, deletion/phase changes, failed ticks, and invalid input invalidate.
    o={8,0,true,true,true,true};auto recycled=advanceSettling(state,o);
    CHECK((recycled==SettlingState{8,1,0,0}));
    o.eligible=false;CHECK((advanceSettling(state,o)==SettlingState{8,0,0,0}));
    o.eligible=true;o.completedTick=false;CHECK(advanceSettling(state,o).quietTicks==0);
    o.completedTick=true;o.generation=0;CHECK(advanceSettling(state,o).quietTicks==0);
    o.generation=7;
    for(float speed:{-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        o.speed=speed;CHECK(!advanceSettling(state,o).candidate && !advanceSettling(state,o).quietTicks);
    }
    o.speed=0;o.holdContact=false;CHECK(rejects([&]{advanceSettling(state,o);}));o.holdContact=true;
    // An interrupted acquisition needs a fresh full dwell, never cumulative quiet ticks.
    state={};
    for(unsigned tick=0;tick<29;++tick)state=advanceSettling(state,o);
    o.enterContact=false;state=advanceSettling(state,o);CHECK(!state.quietTicks);
    o.enterContact=true;
    for(unsigned tick=0;tick<29;++tick)state=advanceSettling(state,o);
    CHECK(!state.candidate);state=advanceSettling(state,o);CHECK(state.candidate);
    // Extreme integer profiles cannot wrap either counter.
    p.acquireTicks=std::numeric_limits<std::uint32_t>::max();p.releaseTicks=p.acquireTicks;
    state={7,p.acquireTicks-1,0,0};state=advanceSettling(state,o,p);CHECK(state.candidate && state.quietTicks==p.acquireTicks);
    state.unsupportedTicks=p.releaseTicks-1;o.enterContact=false;o.holdContact=false;
    CHECK(!advanceSettling(state,o,p).candidate);
    p={};p.acquireTicks=1;p.releaseTicks=1;o.enterContact=true;o.holdContact=true;
    state=advanceSettling({},o,p);CHECK(state.candidate);
    o.enterContact=false;o.holdContact=false;CHECK(!advanceSettling(state,o,p).candidate);
    o.enterContact=true;o.holdContact=true;
    CHECK(advanceSettling({7,999,999,2},o).quietTicks==1);
    for(float bad:{-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        p={};p.enterSpeed=bad;CHECK(rejects([&]{advanceSettling({},o,p);}));
        p={};p.exitSpeed=bad;CHECK(rejects([&]{advanceSettling({},o,p);}));
    }
    p={};p.exitSpeed=p.enterSpeed;CHECK(rejects([&]{advanceSettling({},o,p);}));
    p={};p.acquireTicks=0;CHECK(rejects([&]{advanceSettling({},o,p);}));
    p={};p.releaseTicks=0;CHECK(rejects([&]{advanceSettling({},o,p);}));
    // Geometry is checked independently from state transitions: floor, an upright
    // stack, dead/non-solid support, static support, deep overlap and side contacts.
    using namespace pw::reference;
    std::vector<SettlingBody> bodies{{{0,0.0625,0}},{{0,0.1875,0}},{{0.125,0.1875,0}}};
    CHECK(settlingSupport(bodies,0).enter);
    CHECK(settlingSupport(bodies,1).enter);
    CHECK(!settlingSupport(bodies,2).hold);
    bodies[0].dynamic=false;CHECK(settlingSupport(bodies,1).enter && !settlingSupport(bodies,0).enter);
    bodies[0].alive=false;CHECK(!settlingSupport(bodies,1).hold);
    bodies[0].alive=true;bodies[0].solid=false;CHECK(!settlingSupport(bodies,1).hold);
    bodies[0].solid=true;bodies[1].position[1]=0.1885;
    CHECK(!settlingSupport(bodies,1).enter && settlingSupport(bodies,1).hold);
    bodies[1].position[1]=0.19;CHECK(!settlingSupport(bodies,1).hold);
    bodies[1].position[1]=0.1;CHECK(!settlingSupport(bodies,1).hold);
    bodies[1].position=bodies[0].position;bodies[0].position[1]=8;bodies[1].position[1]=8;
    CHECK(!settlingSupport(bodies,1).hold);
    bodies.resize(1);bodies[0].dynamic=true;bodies[0].position={0,0.0635,0};
    CHECK(!settlingSupport(bodies,0).enter && settlingSupport(bodies,0).hold);
    bodies[0].position[1]=0.065;CHECK(!settlingSupport(bodies,0).hold);
    bodies[0].position[1]=0.05;CHECK(!settlingSupport(bodies,0).hold);
    bodies[0].position[1]=std::numeric_limits<double>::quiet_NaN();CHECK(!settlingSupport(bodies,0).hold);
    std::cout<<"Settling hysteresis: dwell, speed/contact dead bands, recycling, invalidation and counter bounds passed\n";
}
