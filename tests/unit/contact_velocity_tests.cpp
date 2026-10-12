#include "../reference/ContactVelocity.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}} while(false)
using namespace pw::reference;
bool near(double a,double b,double tolerance=1e-10) {return std::abs(a-b)<=tolerance;}
std::vector<ContactVelocityPoint> pair(double restitution,double friction) {
    std::vector<ContactVelocityPoint> p(2);
    for(unsigned i=0;i<2;++i) {p[i].slot=i;p[i].position=p[i].predictedPosition={i*0.125,4,0};p[i].restitution=restitution;p[i].friction=friction;}
    p[0].velocity=p[0].incomingVelocity={1,0,0};p[1].velocity=p[1].incomingVelocity={-1,0,0};return p;
}
double energy(std::span<const ContactVelocityPoint> p,std::span<const VelocityVector> v) {
    double result=0;for(std::size_t i=0;i<p.size();++i) if(p[i].active && p[i].inverseMass>0)
        result+=contact_detail::dot(v[i],v[i])/(2*p[i].inverseMass);return result;
}
VelocityVector momentum(std::span<const ContactVelocityPoint> p,std::span<const VelocityVector> v) {
    VelocityVector result{};for(std::size_t i=0;i<p.size();++i) if(p[i].active && p[i].inverseMass>0)
        for(unsigned axis=0;axis<3;++axis) result[axis]+=v[i][axis]/p[i].inverseMass;return result;
}
int main() {
    auto p=pair(1,0);auto r=contactVelocityResponse(p);
    CHECK(r.pairs==1 && r.degrees==std::vector<std::uint32_t>({1,1}));
    CHECK(near(r.velocities[0][0],-1) && near(r.velocities[1][0],1));
    p=pair(0.5,0);r=contactVelocityResponse(p);CHECK(near(r.velocities[0][0],-0.5) && near(r.velocities[1][0],0.5));
    p=pair(0,0);r=contactVelocityResponse(p);CHECK(near(r.velocities[0][0],0) && near(r.velocities[1][0],0));
    p=pair(1,0);p[0].velocity=p[0].incomingVelocity={0.125,0,0};p[1].velocity=p[1].incomingVelocity={-0.125,0,0};
    r=contactVelocityResponse(p);CHECK(near(r.velocities[0][0],0) && near(r.velocities[1][0],0));
    p=pair(1,0);p[0].velocity=p[0].incomingVelocity={-1,0,0};p[1].velocity=p[1].incomingVelocity={1,0,0};
    r=contactVelocityResponse(p);CHECK(r.velocities[0]==p[0].velocity && r.velocities[1]==p[1].velocity);
    p=pair(1,0);p[1].restitution=0.25;r=contactVelocityResponse(p);CHECK(near(r.velocities[0][0],-0.25) && near(r.velocities[1][0],0.25));
    p=pair(0,1);p[0].friction=0.25;p[0].velocity=p[0].incomingVelocity={1,1,0};p[1].velocity=p[1].incomingVelocity={-1,-1,0};
    r=contactVelocityResponse(p);CHECK(near(r.velocities[0][1],0.5) && near(r.velocities[1][1],-0.5));
    p=pair(0,1);p[0].velocity=p[0].incomingVelocity={1,0.01,0};p[1].velocity=p[1].incomingVelocity={-1,-0.01,0};
    r=contactVelocityResponse(p);CHECK(near(r.velocities[0][1],0) && near(r.velocities[1][1],0));
    p=pair(0.5,0);p[1].inverseMass=0;p[1].velocity=p[1].incomingVelocity={0,0,0};r=contactVelocityResponse(p);
    CHECK(near(r.velocities[0][0],-0.5) && r.velocities[1]==p[1].velocity);
    p[0].inverseMass=0;r=contactVelocityResponse(p);CHECK(r.pairs==0 && r.velocities[0]==p[0].velocity);
    p=pair(0,0);p[0].inverseMass=0.25;r=contactVelocityResponse(p);
    CHECK(near(r.velocities[0][0],0.6) && near(r.velocities[1][0],0.6));
    p=pair(1,0);p[0].incomingVelocity=p[1].incomingVelocity={0,0,0};p[0].velocity={-2,0,0};p[1].velocity={2,0,0};
    p[1].predictedPosition[0]=0.0625;r=contactVelocityResponse(p);CHECK(near(r.velocities[0][0],0) && near(r.velocities[1][0],0));
    // Retain predicted contacts after opposite corrections separate the centres.
    p=pair(0,0);p[0].position[0]=-0.25;p[1].position[0]=0.375;r=contactVelocityResponse(p);CHECK(r.pairs==1);
    const auto distance=contact_detail::subtract(p[0].position,p[1].position);const double support=0.125+2*(0.25+1e-5);
    CHECK(contact_detail::dot(distance,distance)<=support*support);
    // Discover contacts created by corrections, even when prediction had none.
    p=pair(0,0);p[1].predictedPosition[0]=0.25;r=contactVelocityResponse(p);CHECK(r.pairs==1);
    p[1].position[0]=p[1].predictedPosition[0]=0.25;r=contactVelocityResponse(p);CHECK(r.pairs==0);
    p=pair(0,0);p.push_back(p[1]);p[2].slot=2;p[2].position=p[2].predictedPosition={0.25,4,0};
    p[1].incomingVelocity=p[1].velocity={0,0,0};r=contactVelocityResponse(p);
    CHECK(r.degrees==std::vector<std::uint32_t>({1,2,1}));CHECK(near(r.velocities[0][0],0.75) && near(r.velocities[2][0],-0.75));
    // Analytic isolated-pair invariants, including strong friction and low speed.
    unsigned pairCases=0;
    for(double e:{0.,0.25,0.5,1.}) for(double mu:{0.,0.1,0.5,1.}) for(double mass:{0.25,1.,4.}) for(double tangent:{0.,0.001,0.5,4.}) {
        p=pair(e,mu);p[0].inverseMass=mass;p[0].velocity=p[0].incomingVelocity={1,tangent,0};p[1].velocity=p[1].incomingVelocity={-1,-tangent,0};
        r=contactVelocityResponse(p);std::vector<VelocityVector> before{p[0].velocity,p[1].velocity};
        CHECK(energy(p,r.velocities)<=energy(p,before)+1e-10);
        const auto oldMomentum=momentum(p,before),newMomentum=momentum(p,r.velocities);
        for(unsigned axis=0;axis<3;++axis) CHECK(near(oldMomentum[axis],newMomentum[axis]));
        CHECK(r.velocities[0][1]-r.velocities[1][1]>=-1e-12);
        CHECK(r.velocities[0][1]-r.velocities[1][1]<=2*tangent+1e-12);++pairCases;
    }
    std::mt19937 random(7123);unsigned graphCases=0;
    for(unsigned seed=0;seed<128;++seed) {
        p.clear();const unsigned count=2+random()%64;
        for(unsigned i=0;i<count;++i) {
            ContactVelocityPoint body;body.slot=100+17*i;
            body.position=body.predictedPosition={double(random()%9)/64,4+double(random()%9)/64,double(random()%9)/64};
            for(auto& v:body.velocity) v=(int(random()%257)-128)/32.;body.incomingVelocity=body.velocity;
            body.inverseMass=double(1u<<(random()%5))/4;body.restitution=double(random()%5)/4;body.friction=double(random()%5)/4;p.push_back(body);
        }
        const auto original=p;std::vector<VelocityVector> before;for(const auto& body:p) before.push_back(body.velocity);
        r=contactVelocityResponse(p);CHECK(p==original);
        CHECK(energy(p,r.velocities)<=energy(p,before)+1e-8);
        const auto oldMomentum=momentum(p,before),newMomentum=momentum(p,r.velocities);
        for(unsigned axis=0;axis<3;++axis) CHECK(near(oldMomentum[axis],newMomentum[axis],1e-8));
        // Slot IDs, not storage order, define coincident normals and membership.
        std::reverse(p.begin(),p.end());const auto reversed=contactVelocityResponse(p);
        for(unsigned i=0;i<count;++i) for(unsigned axis=0;axis<3;++axis) CHECK(near(r.velocities[i][axis],reversed.velocities[count-1-i][axis],1e-10));
        ++graphCases;
    }
    p=pair(0,0);p[1].active=false;p[1].position[0]=std::numeric_limits<double>::quiet_NaN();r=contactVelocityResponse(p);
    CHECK(r.pairs==0 && r.velocities[0]==p[0].velocity && r.velocities[1]==p[1].velocity);
    auto invalid=[](auto f){try {f();return false;} catch(const std::invalid_argument&){return true;}};
    p=pair(0,0);p[1].slot=p[0].slot;CHECK(invalid([&]{contactVelocityResponse(p);}));
    p=pair(0,0);p[0].restitution=1.01;CHECK(invalid([&]{contactVelocityResponse(p);}));
    p=pair(0,0);p[0].friction=-1;CHECK(invalid([&]{contactVelocityResponse(p);}));
    p=pair(0,0);p[0].position[0]=-0.5;CHECK(invalid([&]{contactVelocityResponse(p);}));
    p=pair(0,0);p[0].velocity[0]=std::numeric_limits<double>::infinity();CHECK(invalid([&]{contactVelocityResponse(p);}));
    p=pair(0,0);p[0].inverseMass=std::numeric_limits<float>::denorm_min();CHECK(invalid([&]{contactVelocityResponse(p);}));
    CHECK(contactVelocityResponse({}).velocities.empty());
    std::cout<<"Contact velocity reference passed: "<<pairCases<<" pair invariants, "<<graphCases<<" immutable graph energy/momentum/permutation cases and targeted fixtures\n";
}
