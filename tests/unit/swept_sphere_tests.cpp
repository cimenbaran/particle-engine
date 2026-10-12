#include "../reference/SweptSphere.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
using namespace pw::reference;
bool near(double a,double b,double tolerance=1e-10){return std::abs(a-b)<=tolerance*std::max({1.0,std::abs(a),std::abs(b)});}
double speed2(const SweepState& s){double n=0;for(double v:s.velocity)n+=v*v;return n;}
template<class F>bool rejects(F f){try{f();return false;}catch(const std::exception&){return true;}}
int main(){
    const SphereCollider sphere{{0,0,0},0.9375};const double radius=0.0625;
    const SweepState input{{-2,0,0},{10,0,0}};
    auto r=sweepSphere(input,0.2,radius,sphere,0.5,0);
    CHECK(r.hit && near(r.time,0.1) && near(r.state.position[0],-1.5) && near(r.state.velocity[0],-5));
    CHECK(r.normal==SweepVector({-1,0,0}) && input.position[0]==-2);
    r=sweepSphere(input,0.1,radius,sphere,1,0);CHECK(r.hit && near(r.state.position[0],-1) && r.state.velocity[0]==-10);
    r=sweepSphere(input,0.05,radius,sphere,1,0);CHECK(!r.hit && near(r.state.position[0],-1.5));
    r=sweepSphere(input,0,radius,sphere,1,0);CHECK(!r.hit && r.state==input);
    r=sweepSphere({{-2,0,0},{-10,0,0}},0.2,radius,sphere,1,0);CHECK(!r.hit && r.state.position[0]==-4);
    r=sweepSphere({{-1,0,0},{0.1,0,0}},0.2,radius,sphere,1,0);CHECK(r.hit && r.time==0 && r.state.velocity==SweepVector({0,0,0}));
    r=sweepSphere({{-1,0,0},{0,1,0}},0.2,radius,sphere,1,1);CHECK(!r.hit && near(r.state.position[1],0.2));
    r=sweepSphere({{-2,1,0},{10,0,0}},0.4,radius,sphere,1,1);CHECK(!r.hit && near(r.state.position[0],2));
    r=sweepSphere({{-2,1.01,0},{10,0,0}},0.4,radius,sphere,1,1);CHECK(!r.hit);
    // Exact surface contact separates normal and tangent response analytically.
    r=sweepSphere({{-1,0,0},{2,3,0}},0.1,radius,sphere,0.5,0.25,0);
    CHECK(r.hit && near(r.state.velocity[0],-1) && near(r.state.velocity[1],2.25));
    r=sweepSphere({{-1,0,0},{2,1,0}},0.1,radius,sphere,0,1,0);
    CHECK(r.state.velocity==SweepVector({0,0,0}));
    // Fast shot crosses the entire obstacle within one tick; endpoint overlap
    // alone would miss it. Near-surface shot exercises the stable small root.
    r=sweepSphere({{-4,0,0},{10000,0,0}},0.01,radius,sphere,0,0);
    CHECK(r.hit && near(r.time,0.0003) && near(r.state.position[0],-1));
    const double outside=std::nextafter(1.0,2.0);
    r=sweepSphere({{-outside,0,0},{1,0,0}},1,radius,sphere,1,0,0);
    CHECK(r.hit && near(r.time,outside-1,1e-15));
    r=sweepSphere({{-1e20,0,0},{1e20,0,0}},1,radius,sphere,0,0);
    CHECK(r.hit && near(r.state.position[0],-1) && r.state.velocity[0]==0);
    const double tiny=std::numeric_limits<float>::min();
    r=sweepSphere({{-4*tiny,0,0},{8*tiny,0,0}},0.5,tiny,{{0,0,0},tiny},0,0,0);
    CHECK(r.hit && near(r.time,0.25) && r.state.position[0]==-2*tiny);
    std::mt19937 random(20261014);std::uniform_real_distribution<double> offset(0,0.99),speed(5,100),length(2,4);
    unsigned cases=0;
    for(unsigned i=0;i<512;++i){
        const double impact=offset(random),v=speed(random),start=length(random);
        const double hitX=-std::sqrt(1-impact*impact),expectedTime=(start+hitX)/v,dt=expectedTime+0.01;
        SweepState s{{-start,impact,0},{v,0,0}};
        for(double restitution:{0.0,0.5,1.0})for(double friction:{0.0,0.5,1.0}){
            r=sweepSphere(s,dt,radius,sphere,restitution,friction,0);
            CHECK(r.hit && near(r.time,expectedTime));CHECK(near(r.normal[0],hitX) && near(r.normal[1],impact));
            CHECK(speed2(r.state)<=speed2(s)+1e-8);
            CHECK(std::hypot(r.state.position[0],r.state.position[1],r.state.position[2])>=1-1e-12);
            CHECK(near(std::hypot(r.normal[0],r.normal[1],r.normal[2]),1));
            // Independent signed axis permutation and translation covariance.
            SweepState mapped{{3+s.position[2],4-s.position[0],-2+s.position[1]}, {s.velocity[2],-s.velocity[0],s.velocity[1]}};
            auto transformed=sweepSphere(mapped,dt,radius,{{3,4,-2},sphere.radius},restitution,friction,0);
            CHECK(transformed.hit && near(transformed.time,r.time));
            CHECK(near(transformed.state.position[0],3+r.state.position[2]) && near(transformed.state.position[1],4-r.state.position[0]) && near(transformed.state.position[2],-2+r.state.position[1]));
            CHECK(near(transformed.state.velocity[0],r.state.velocity[2]) && near(transformed.state.velocity[1],-r.state.velocity[0]) && near(transformed.state.velocity[2],r.state.velocity[1]));
            if(restitution==1 && friction==0){
                SweepState reverse=r.state;for(double& component:reverse.velocity)component=-component;
                auto back=sweepSphere(reverse,dt,radius,sphere,1,0,0);
                CHECK(back.hit);for(unsigned axis=0;axis<3;++axis){CHECK(near(back.state.position[axis],s.position[axis]));CHECK(near(back.state.velocity[axis],-s.velocity[axis]));}
            }
            ++cases;
        }
    }
    CHECK(rejects([&]{sweepSphere({{0,0,0},{0,0,0}},0,radius,sphere,0,0);}));
    for(double bad:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
        CHECK(rejects([&]{sweepSphere(input,bad,radius,sphere,0,0);}));
        CHECK(rejects([&]{sweepSphere(input,0.1,bad,sphere,0,0);}));
        CHECK(rejects([&]{sweepSphere(input,0.1,radius,{{0,0,0},bad},0,0);}));
        CHECK(rejects([&]{sweepSphere(input,0.1,radius,sphere,bad,0);}));
        CHECK(rejects([&]{sweepSphere(input,0.1,radius,sphere,0,bad);}));
        CHECK(rejects([&]{sweepSphere(input,0.1,radius,sphere,0,0,bad);}));
    }
    CHECK(rejects([&]{sweepSphere(input,0.1,0,sphere,0,0);}));
    CHECK(rejects([&]{sweepSphere(input,0.1,radius,{{0,0,0},0},0,0);}));
    CHECK(rejects([&]{sweepSphere(input,1.1,radius,sphere,0,0);}));
    CHECK(rejects([&]{sweepSphere(input,0.1,radius,sphere,1.1,0);}));
    CHECK(rejects([&]{sweepSphere(input,0.1,radius,sphere,0,1.1);}));
    for(unsigned axis=0;axis<3;++axis){
        auto invalid=input;invalid.position[axis]=std::numeric_limits<double>::quiet_NaN();CHECK(rejects([&]{sweepSphere(invalid,0.1,radius,sphere,0,0);}));
        invalid=input;invalid.velocity[axis]=std::numeric_limits<double>::infinity();CHECK(rejects([&]{sweepSphere(invalid,0.1,radius,sphere,0,0);}));
        auto bad=sphere;bad.centre[axis]=std::numeric_limits<double>::infinity();CHECK(rejects([&]{sweepSphere(input,0.1,radius,bad,0,0);}));
    }
    std::cout<<"Swept sphere: analytic entry/tangent/endpoint/fast shots, "<<cases<<" dissipative and transformed cases, elastic reversal and invalid inputs passed\n";
}
