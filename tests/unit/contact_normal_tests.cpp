#include "../reference/ContactNormal.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
int main() {
    using namespace pw::reference;
    using V=std::array<double,3>;
    auto close=[](double a,double b){return std::abs(a-b)<1e-12;};
    std::vector<NormalBody> bodies(1);bodies[0].position={0,8,0};
    CHECK(contactNormal(bodies,0).source==NormalSource::None);
    for(unsigned axis=0;axis<3;++axis)for(unsigned side=0;side<2;++side) {
        bodies[0].position={0,8,0};
        bodies[0].position[axis]=axis==1?(side?15.9375:0.0625):(side?7.9375:-7.9375);
        auto n=contactNormal(bodies,0);V expected{};expected[axis]=side?-1:1;
        CHECK(n.source==NormalSource::Plane && n.feature==axis*2+side && n.direction==expected && n.gap==0);
    }
    bodies[0].position={-7.9375,0.0625,0};
    CHECK(contactNormal(bodies,0).feature==0); // Exact corner tie, x before y.
    bodies[0].position={0,8,0};bodies.resize(3);
    bodies[1].position={0.125,8,0};bodies[2].position={-0.125,8,0};
    auto n=contactNormal(bodies,0);
    CHECK(n.source==NormalSource::Particle && n.feature==1 && (n.direction==V{-1,0,0}));
    bodies[2].position[0]=-0.12;n=contactNormal(bodies,0);
    CHECK(n.feature==2 && n.gap<0 && (n.direction==V{1,0,0}));
    bodies[2].alive=false;CHECK(contactNormal(bodies,0).feature==1);
    bodies[1].solid=false;CHECK(contactNormal(bodies,0).source==NormalSource::None);
    bodies[1].solid=true;bodies[1].dynamic=false;CHECK(contactNormal(bodies,0).feature==1);
    bodies[0].dynamic=false;CHECK(contactNormal(bodies,0).source==NormalSource::None);bodies[0].dynamic=true;
    bodies.resize(2);bodies[1].position={0.127,8,0};
    // Avoid treating decimal roundoff at 0.002 as a changed contact policy.
    bodies[1].position[0]=std::nextafter(0.127,0.0);CHECK(contactNormal(bodies,0).source==NormalSource::Particle);
    bodies[1].position[0]=std::nextafter(0.127,1.0);CHECK(contactNormal(bodies,0).source==NormalSource::None);
    bodies[1].dynamic=true;bodies[1].position=bodies[0].position;
    auto a=contactNormal(bodies,0),b=contactNormal(bodies,1);
    CHECK(a.gap==-0.125 && b.gap==-0.125);
    for(unsigned axis=0;axis<3;++axis)CHECK(a.direction[axis]==-b.direction[axis]);
    CHECK(close(std::hypot(a.direction[0],a.direction[1],a.direction[2]),1));
    bodies[0].position={0,0.0625,0};bodies[1].position={0,0.1875,0};
    n=contactNormal(bodies,0);CHECK(n.source==NormalSource::Plane && n.feature==2);
    pw::AnalyticSphere sphere{{0,8,0},0.9375f};
    bodies[0].position={1,8,0};bodies[1].position={1.125,8,0};
    n=contactNormal(bodies,0,sphere);CHECK(n.source==NormalSource::Sphere && (n.direction==V{1,0,0}));
    bodies.resize(1);
    for(unsigned mask=0;mask<8;++mask) {
        V direction{};for(unsigned axis=0;axis<3;++axis)direction[axis]=(mask&(1u<<axis)?1:-1)/std::sqrt(3.0);
        for(unsigned axis=0;axis<3;++axis)bodies[0].position[axis]=sphere.centre[axis]+direction[axis];
        n=contactNormal(bodies,0,sphere);CHECK(n.source==NormalSource::Sphere && close(n.gap,0));
        for(unsigned axis=0;axis<3;++axis)CHECK(close(n.direction[axis],direction[axis]));
    }
    bodies[0].position={0,8,0};bool rejected=false;
    try{contactNormal(bodies,0,sphere);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    bodies[0].position[0]=std::numeric_limits<double>::quiet_NaN();rejected=false;
    try{contactNormal(bodies,0);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    bodies[0].alive=false;CHECK(contactNormal(bodies,0).source==NormalSource::None);
    std::cout<<"Contact normal oracle: planes, ties, signed sphere normals, pair exclusions, coincidence and invalid geometry passed\n";
}
