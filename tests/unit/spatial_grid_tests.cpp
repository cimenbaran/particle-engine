#include "core/SpatialGrid.hpp"
#include <cstdlib>
#include <iostream>
#include <set>
#include <limits>
#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}} while(false)
int main() {
    using Grid=pw::SpatialGrid;
    const float inf=std::numeric_limits<float>::infinity();
    const float nan=std::numeric_limits<float>::quiet_NaN();
    Grid grid({0,0,0},0.125f,{-2,-3,-4},{3,4,5});
    CHECK(grid.cellCount()==60);
    CHECK(grid.coordinates({-0.01f,0,0})==Grid::Cell({-1,0,0}));
    CHECK(grid.coordinates({-0.125f,0.125f,0})==Grid::Cell({-1,1,0}));
    CHECK(grid.coordinates({std::nextafter(-0.125f,-inf),0,0})==Grid::Cell({-2,0,0}));
    CHECK(grid.coordinates({std::nextafter(0.125f,0.f),0,0})==Grid::Cell({0,0,0}));
    CHECK(grid.key({-2,-3,-4})==0 && grid.key({0,0,0})==59);
    CHECK(grid.key({-1,-2,-3})==16); // x + 3*y + 12*z, relative to minimum.
    CHECK(!grid.key({1,0,0}) && !grid.key({-3,0,0}) && !grid.decode(60));
    CHECK(!grid.keyAt({0.125f,0,0})); // Upper domain face is excluded.
    CHECK(grid.keyAt({-0.25f,-0.375f,-0.5f})==0); // Lower face included.
    CHECK(!grid.coordinates({nan,0,0}) && !grid.coordinates({0,inf,0}));
    CHECK(!grid.coordinates({std::numeric_limits<float>::max(),0,0}));
    CHECK(!grid.decode(Grid::invalidKey));
    std::set<std::uint32_t> keys;
    for(int z=-4;z<=0;++z) for(int y=-3;y<=0;++y) for(int x=-2;x<=0;++x) {
        const Grid::Cell cell{x,y,z};const auto key=grid.key(cell);
        CHECK(key && keys.insert(*key).second && grid.decode(*key)==cell);
    }
    CHECK(keys.size()==60 && *keys.begin()==0 && *keys.rbegin()==59);
    Grid shifted({10,-20,30},0.5f,{-10,-10,-10},{20,20,20});
    CHECK(shifted.coordinates({9.75f,-20.5f,30.75f})==Grid::Cell({-1,-1,1}));
    CHECK(grid.searchRadius(0)==0 && grid.searchRadius(0.1f)==1);
    CHECK(grid.searchRadius(0.125f)==1 && grid.searchRadius(std::nextafter(0.125f,inf))==2);
    CHECK(grid.searchRadius(0.3f)==3);
    Grid largest({0,0,0},1,{INT32_MIN,0,0},{UINT32_MAX,1,1});
    CHECK(largest.cellCount()==UINT32_MAX && largest.key({INT32_MAX-1,0,0})==UINT32_MAX-1);
    CHECK(largest.decode(UINT32_MAX-1)==Grid::Cell({INT32_MAX-1,0,0}));
    auto invalid=[](auto function) {try {function();return false;} catch(const std::invalid_argument&) {return true;}};
    for(float width:{0.f,-1.f,inf,nan}) CHECK(invalid([&]{Grid({0,0,0},width,{0,0,0},{1,1,1});}));
    CHECK(invalid([&]{Grid({nan,0,0},1,{0,0,0},{1,1,1});}));
    CHECK(invalid([&]{Grid({0,0,0},1,{0,0,0},{0,1,1});}));
    CHECK(invalid([&]{Grid({0,0,0},1,{INT32_MAX,0,0},{2,1,1});}));
    CHECK(invalid([&]{Grid({0,0,0},1,{0,0,0},{65536,65536,1});}));
    CHECK(invalid([&]{Grid({0,0,0},1,{INT32_MIN,INT32_MIN,INT32_MIN},{UINT32_MAX,UINT32_MAX,UINT32_MAX});}));
    for(float support:{-1.f,inf,nan,std::numeric_limits<float>::max()})
        CHECK(invalid([&]{grid.searchRadius(support);}));
    std::cout<<"Spatial grid signed coordinates, keys, boundaries, support and overflow passed\n";
}
