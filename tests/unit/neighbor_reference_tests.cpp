#include "../reference/SpatialIndex.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}} while(false)
using pw::reference::SpatialPoint;
// Independent all-pairs oracle: deliberately never consults cells, keys or ranges.
std::vector<std::uint32_t> bruteForce(std::span<const SpatialPoint> points,const SpatialPoint& query,float support) {
    std::vector<std::uint32_t> result;
    if(!query.active || !query.physical) return result;
    for(const auto& point:points) {
        if(!point.active || !point.physical || point.slot==query.slot || point.interactionClass!=query.interactionClass) continue;
        const double x=double(point.position[0])-query.position[0];
        const double y=double(point.position[1])-query.position[1];
        const double z=double(point.position[2])-query.position[2];
        if(x*x+y*y+z*z<=double(support)*support) result.push_back(point.slot);
    }
    std::sort(result.begin(),result.end());return result;
}
int main() {
    using pw::SpatialGrid;using pw::reference::NeighborReference;
    SpatialGrid grid({0,0,0},0.125f,{-8,-8,-8},{16,16,16});
    const float beyond=std::nextafter(0.125f,1.f);
    std::vector<SpatialPoint> points{
        {100,{0,0,0}}, {90,{0.125f,0,0}}, {80,{-0.125f,0,0}},
        {70,{beyond,0,0}}, {60,{0,0,0}}, {50,{0,0,0},false,true},
        {40,{0,0,0},true,false}, {30,{0,0,0},true,true,1}, {20,{0.124f,0.124f,0}}};
    NeighborReference reference(grid,points);
    CHECK(reference.neighbors(100,0)==std::vector<std::uint32_t>({60}));
    CHECK(reference.neighbors(100,0.125f)==std::vector<std::uint32_t>({60,80,90}));
    CHECK(reference.neighbors(50,1).empty() && reference.neighbors(40,1).empty());
    CHECK(reference.neighbors(30,1).empty());
    // Caller changes cannot silently pair new positions with stale cell ranges.
    points[1].position={0.9f,0,0};
    CHECK(reference.neighbors(100,0.125f)==std::vector<std::uint32_t>({60,80,90}));
    NeighborReference rebuilt(grid,points);
    CHECK(rebuilt.neighbors(100,0.125f)==std::vector<std::uint32_t>({60,80}));
    auto invalid=[](auto function) {try {function();return false;} catch(const std::invalid_argument&) {return true;}};
    CHECK(invalid([&]{reference.neighbors(999,0.1f);}));
    CHECK(invalid([&]{reference.neighbors(100,-1);}));
    CHECK(invalid([&]{reference.neighbors(100,std::numeric_limits<float>::quiet_NaN());}));
    std::mt19937 random(42);
    points.clear();
    for(unsigned i=0;i<257;++i) {
        SpatialGrid::Position position;
        // Explicit integer-derived floats avoid distribution implementation differences.
        for(auto& axis:position) axis=(int(random()%2048)-1024)/1024.f;
        points.push_back({9000+17*i,position,i%13!=0,i%17!=0,i%3});
    }
    for(float width:{0.0625f,0.125f,0.3f}) {
        SpatialGrid configured({0,0,0},width,{-32,-32,-32},{64,64,64});
        NeighborReference indexed(configured,points);
        for(float support:{0.f,0.0625f,0.125f,0.2501f,0.9f}) for(const auto& point:points)
            CHECK(indexed.neighbors(point.slot,support)==bruteForce(points,point,support));
    }
    // No hidden neighbor cap: coincident dense cells preserve every eligible ID.
    points.clear();for(unsigned i=0;i<1024;++i) points.push_back({i,{0,0,0}});
    NeighborReference dense(grid,points);
    auto neighbors=dense.neighbors(512,0);
    CHECK(neighbors.size()==1023 && neighbors.front()==0 && neighbors.back()==1023);
    CHECK(!std::binary_search(neighbors.begin(),neighbors.end(),512));
    std::cout<<"CPU neighbor membership matches brute force; self/class/activity filters and snapshot ownership passed\n";
}
