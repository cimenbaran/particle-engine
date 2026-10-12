#include "../reference/PositionFriction.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
using namespace pw::reference;
bool near(double a,double b,double tolerance=1e-12){return std::abs(a-b)<=tolerance;}
template<class F>bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    const FrictionVector up{0,1,0};
    auto correction=positionFriction({0.001,-0.01,0},up,0.01,0.5,0.3);
    CHECK(near(correction[0],-0.001) && correction[1]==0 && correction[2]==0);
    correction=positionFriction({0.02,-0.01,0},up,0.01,0.5,0.3);
    CHECK(near(correction[0],-0.003));
    correction=positionFriction({0.005,-0.01,0},up,0.01,0.5,0.3);
    CHECK(near(correction[0],-0.005)); // inclusive static threshold
    correction=positionFriction({std::nextafter(0.005,1.0),-0.01,0},up,0.01,0.5,0.3);
    CHECK(near(correction[0],-0.003));
    CHECK((positionFriction({1,2,3},up,0,1,1)==FrictionVector{}));
    CHECK((positionFriction({1,2,3},up,1,0,0)==FrictionVector{}));
    CHECK((positionFriction({0,-1,0},up,1,1,1)==FrictionVector{}));
    auto pair=positionFrictionPair({0.02,0,0},up,0.01,1,3,0.5,0.3);
    CHECK(near(pair.a[0],-0.00075) && near(pair.b[0],0.00225));
    pair=positionFrictionPair({0.02,0,0},up,0.01,1,0,0.5,0.3);
    CHECK(near(pair.a[0],-0.003) && pair.b==FrictionVector{});
    pair=positionFrictionPair({0.02,0,0},up,0.01,0,0,0.5,0.3);CHECK(pair.a==FrictionVector{} && pair.b==FrictionVector{});
    // Independently known inclined-plane solution: tan(theta)<mu supports rest.
    // Cancelling velocity after projection still leaves positional drift. Applying
    // the tangential displacement constraint removes that drift without sleeping.
    const double angle=0.2,g=9.81,h=1.0/240,mu=0.5;
    const FrictionVector normal{-std::sin(angle),std::cos(angle),0};
    const FrictionVector gravityStep{0,-g*h*h,0};const double depth=-frictionDot(gravityStep,normal);
    FrictionVector projected{},velocityOnly{},positionLevel{};
    for(unsigned axis=0;axis<3;++axis)projected[axis]=gravityStep[axis]+depth*normal[axis];
    CHECK(std::tan(angle)<mu);
    const auto sticking=positionFriction(projected,normal,depth,mu,mu);
    for(unsigned tick=0;tick<10000;++tick)for(unsigned axis=0;axis<3;++axis){velocityOnly[axis]+=projected[axis];positionLevel[axis]+=projected[axis]+sticking[axis];}
    const double expectedDrift=10000*g*h*h*std::sin(angle);
    CHECK(near(std::hypot(velocityOnly[0],velocityOnly[1],velocityOnly[2]),expectedDrift,1e-10));
    CHECK(std::hypot(positionLevel[0],positionLevel[1],positionLevel[2])<1e-12);
    std::mt19937 random(20261013);std::uniform_real_distribution<double> component(-1,1),unit(0,1);
    for(unsigned trial=0;trial<1024;++trial){
        FrictionVector n{component(random),component(random),component(random)},d{component(random),component(random),component(random)};
        const double magnitude=std::hypot(n[0],n[1],n[2]);for(auto& value:n)value/=magnitude;
        const double support=unit(random),ms=unit(random),mk=ms*unit(random),a=0.1+unit(random),b=0.1+unit(random);
        pair=positionFrictionPair(d,n,support,a,b,ms,mk);FrictionVector tangent{},remaining{};const auto along=frictionDot(d,n);
        for(unsigned axis=0;axis<3;++axis){
            CHECK(near(pair.a[axis]/a+pair.b[axis]/b,0)); // unchanged mass-weighted centre
            tangent[axis]=d[axis]-along*n[axis];remaining[axis]=tangent[axis]+pair.a[axis]-pair.b[axis];
        }
        const double before=std::hypot(tangent[0],tangent[1],tangent[2]),after=std::hypot(remaining[0],remaining[1],remaining[2]);
        CHECK(after<=before+1e-12 && frictionDot(tangent,remaining)>=-1e-12);
        CHECK(near(frictionDot(pair.a,n),0) && near(frictionDot(pair.b,n),0));
        if(before<=ms*support)CHECK(after<1e-12);else CHECK(near(before-after,std::min(before,mk*support)));
        FrictionVector reversed{};for(unsigned axis=0;axis<3;++axis)reversed[axis]=-d[axis];
        auto swapped=positionFrictionPair(reversed,n,support,b,a,ms,mk);
        for(unsigned axis=0;axis<3;++axis){CHECK(near(pair.a[axis],swapped.b[axis]));CHECK(near(pair.b[axis],swapped.a[axis]));}
    }
    for(double bad:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        CHECK(rejects([&]{positionFriction({1,0,0},up,bad,0.5,0.3);}));
        CHECK(rejects([&]{positionFriction({1,0,0},up,1,bad,0.3);}));
        CHECK(rejects([&]{positionFriction({1,0,0},up,1,0.5,bad);}));
        CHECK(rejects([&]{positionFrictionPair({1,0,0},up,1,bad,1,0.5,0.3);}));
    }
    CHECK(rejects([&]{positionFriction({1,0,0},{0,2,0},1,0.5,0.3);}));
    CHECK(rejects([&]{positionFriction({1,0,0},up,1,0.3,0.5);}));
    CHECK(rejects([&]{positionFriction({1,0,0},up,1,1.1,0.5);}));
    std::cout<<"Position friction: analytic traction, 10000-step incline drift regression and 1024 pair invariants passed; velocity-only drift="<<expectedDrift<<" m\n";
}
