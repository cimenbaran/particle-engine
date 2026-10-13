#include "../reference/SweptVoxel.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
using namespace pw::reference;
bool near(double a,double b,double tolerance=1e-10){return std::abs(a-b)<=tolerance;}
template<class F>bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    constexpr double r=0.0625;
    for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
        VoxelVector start{r,r,r},end=start;start[axis]=sign<0?-1:1;end[axis]=sign<0?1:-1;
        auto hit=sweepVoxelSphere(start,end,{});CHECK(hit.hit && !hit.initialOverlap);
        CHECK(near(hit.time,sign<0?(1-r)/2:(1-0.125-r)/2));
        for(unsigned a=0;a<3;++a)CHECK(near(hit.normal[a],a==axis?sign:0));
    }
    auto edge=sweepVoxelSphere({-1,-r*0.6,r},{1,-r*0.6,r},{});
    CHECK(edge.hit && near(edge.time,(1-r*0.8)/2));
    CHECK(near(edge.normal[0],-0.8) && near(edge.normal[1],-0.6) && edge.normal[2]==0);
    auto corner=sweepVoxelSphere({-1,-r/3,-r/3},{1,-r/3,-r/3},{});
    CHECK(corner.hit && near(corner.time,(1-r*std::sqrt(7.0)/3)/2));
    CHECK(near(corner.normal[0],-std::sqrt(7.0)/3) && near(corner.normal[1],-1.0/3) && near(corner.normal[2],-1.0/3));
    // A ray intersects the expanded AABB but misses the actual rounded corner.
    CHECK(!sweepVoxelSphere({-1,-r*0.9,-r*0.9},{1,-r*0.9,-r*0.9},{}).hit);
    auto tangent=sweepVoxelSphere({-1,-r,r},{1,-r,r},{});
    CHECK(tangent.hit && near(tangent.time,0.5) && near(tangent.normal[1],-1));
    CHECK(!sweepVoxelSphere({-1,std::nextafter(-r,-1.0),r},{1,std::nextafter(-r,-1.0),r},{}).hit);
    CHECK(sweepVoxelSphere({-1,r,r},{-r,r,r},{}).hit);
    CHECK(!sweepVoxelSphere({-1,r,r},{std::nextafter(-r,-1.0),r,r},{}).hit);
    auto touching=sweepVoxelSphere({-r,r,r},{-1,r,r},{});CHECK(touching.hit && !touching.initialOverlap && touching.time==0);
    auto inside=sweepVoxelSphere({r,r,r},{r,r,r},{});CHECK(inside.hit && inside.initialOverlap && inside.time==0 && inside.normal[0]==-1);
    CHECK(!sweepVoxelSphere({1,1,1},{1,1,1},{}).hit);
    CHECK(near(sweepVoxelSphere({-1000,r,r},{1000,r,r},{}).time,(1000-r)/2000));
    CHECK(rejects([&]{sweepVoxelSphere({},{},{},0);}));
    CHECK(rejects([&]{sweepVoxelSphere({},{},{},5);}));
    for(double invalid:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),1025.0})
        CHECK(rejects([&]{sweepVoxelSphere({invalid,0,0},{},{ });}));
    // Independent convex minimization plus bisection uses distance samples,
    // without plane intervals or quadratic roots from the implementation.
    std::mt19937 random(49071);auto coordinate=[&]{return double(random()%1000001)/1000000-0.5;};
    unsigned compared=0,hits=0;
    for(unsigned test=0;test<4096;++test) {
        VoxelVector start,end;for(unsigned axis=0;axis<3;++axis){start[axis]=coordinate();end[axis]=coordinate();}
        const auto distance=[&](double t) {
            double squared=0;
            for(unsigned axis=0;axis<3;++axis){double p=start[axis]*(1-t)+end[axis]*t;double d=p<0?-p:p>0.125?p-0.125:0;squared+=d*d;}
            return std::sqrt(squared);
        };
        double low=0,high=1;
        for(unsigned step=0;step<160;++step){double a=(2*low+high)/3,b=(low+2*high)/3;if(distance(a)<distance(b))high=b;else low=a;}
        const double minimum=(low+high)/2,gap=distance(minimum)-r;auto result=sweepVoxelSphere(start,end,{});
        if(std::abs(gap)<1e-10)continue;
        ++compared;CHECK(result.hit==(gap<0));
        if(!result.hit)continue;
        ++hits;double expected=0;
        if(distance(0)>r) {
            low=0;high=minimum;for(unsigned step=0;step<100;++step){double middle=(low+high)/2;if(distance(middle)>r)low=middle;else high=middle;}
            expected=(low+high)/2;
        }
        CHECK(near(result.time,expected,1e-9));
        CHECK(near(std::hypot(result.normal[0],result.normal[1],result.normal[2]),1));
        if(!result.initialOverlap) {
            CHECK(near(distance(result.time),r));
            double separation=0;
            for(unsigned axis=0;axis<3;++axis) {
                CHECK(result.point[axis]>=0 && result.point[axis]<=0.125);
                const double d=result.centre[axis]-result.point[axis];separation+=d*d;
                CHECK(near(d,result.normal[axis]*r));
            }
            CHECK(near(std::sqrt(separation),r));
        }
        VoxelVector shiftedStart=start,shiftedEnd=end,origin{-4,8,-12};
        for(unsigned axis=0;axis<3;++axis){shiftedStart[axis]+=origin[axis];shiftedEnd[axis]+=origin[axis];}
        auto shifted=sweepVoxelSphere(shiftedStart,shiftedEnd,origin);CHECK(shifted.hit && near(shifted.time,result.time,1e-9));
    }
    CHECK(compared>4000 && hits>100);
    std::cout<<"Swept voxel faces/edges/corners and "<<compared<<" independent segment cases passed; hits "<<hits<<"\n";
}
