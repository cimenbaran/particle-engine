#include "core/SpawnPlacement.hpp"
#include <algorithm>
#include <iostream>
#define CHECK(v) do{if(!(v))throw std::runtime_error("Spawn placement check failed: " #v);}while(false)
#include <limits>
#include <random>
#include <set>
using namespace pw;
int main() {
    auto verify=[](std::array<float,3> lo,std::array<float,3> hi,std::uint32_t seed) {
        const auto lattice=spawnLattice(lo,hi);
        std::array<std::uint32_t,3> counts{};
        for(unsigned a=0;a<3;++a)for(int cell=-512;cell<=512;++cell)
            counts[a]+=float(cell)*.125f>=lo[a] && float(cell)*.125f<=hi[a];
        CHECK(lattice.sites==std::uint64_t(counts[0])*counts[1]*counts[2]);
        std::set<std::array<float,3>> positions;
        for(std::uint32_t request=0;request<std::min(4096u,lattice.sites);++request) {
            auto p=spawnLatticePosition(lattice,request,seed);CHECK(p);
            for(unsigned a=0;a<3;++a){CHECK((*p)[a]>=lo[a] && (*p)[a]<=hi[a]);CHECK(std::floor((*p)[a]*8)==(*p)[a]*8);}
            CHECK(positions.insert(*p).second);
        }
        // Any distinct global lattice sites are separated by at least .125m.
        CHECK(!spawnLatticePosition(lattice,lattice.sites,seed));
        return positions;
    };
    CHECK(verify({0,0,0},{0,0,0},0).size()==1);
    CHECK(verify({.01f,.01f,.01f},{.1f,.1f,.1f},0).empty());
    auto a=verify({-1,2,-1},{1,3,1},3),b=verify({-1,2,-1},{1,3,1},987);
    CHECK(a==b);
    for(auto i=a.begin();i!=a.end();++i)for(auto j=a.begin();j!=i;++j) {
        double distance=0;for(unsigned axis=0;axis<3;++axis){double d=double((*i)[axis])-(*j)[axis];distance+=d*d;}
        CHECK(distance>=.125*.125);
    }
    const float tiny=std::numeric_limits<float>::denorm_min();
    CHECK(verify({tiny,0,0},{.125f,0,0},0).size()==1);
    CHECK(verify({-.125f,0,0},{-tiny,0,0},0).size()==1);
    CHECK(verify({-0.0f,0,0},{0,0,0},0).size()==1);
    verify({std::nextafter(-.125f,0.0f),0,0},{std::nextafter(.125f,0.0f),0,0},1);
    auto maximum=spawnLattice({-64,-64,-64},{64,64,64});CHECK(maximum.sites==1076890625u);
    verify({-64,-64,-64},{64,64,64},UINT32_MAX);
    std::mt19937 random(55123);
    for(unsigned test=0;test<256;++test) {
        std::array<float,3> lo{},hi{};
        for(unsigned axis=0;axis<3;++axis){lo[axis]=float(int(random()%1025)-512)*.03125f;hi[axis]=lo[axis]+float(random()%17)*.0625f;}
        verify(lo,hi,random());
    }
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),65.0f}) {
        bool rejected=false;try{spawnLattice({0,0,0},{bad,1,1});}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    }
    bool rejected=false;try{spawnLattice({1,0,0},{0,1,1});}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    auto invalid=spawnLattice({0,0,0},{0,0,0});invalid.dimensions[0]=0;
    rejected=false;try{spawnLatticePosition(invalid,0,0);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    invalid=spawnLattice({0,0,0},{0,0,0});invalid.sites=2;
    rejected=false;try{spawnLatticePosition(invalid,0,0);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    std::cout<<"Spawn placement: closed bounds, spacing, seed permutations, subnormals, maximum domain and 256 randomized brushes passed\n";
}
