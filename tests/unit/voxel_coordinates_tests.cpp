#include "core/VoxelCoordinates.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
template<class F> bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    using namespace pw::terrain;
    CHECK(chunkMetres==chunkSide*voxelMetres && voxelsPerChunk==32768);
    for(unsigned z=0;z<32;++z)for(unsigned y=0;y<32;++y)for(unsigned x=0;x<32;++x) {
        const LocalVoxel local{std::uint8_t(x),std::uint8_t(y),std::uint8_t(z)};
        auto index=localIndex(local);CHECK(index && *index==x+32*y+1024*z && decodeLocal(*index)==local);
    }
    CHECK(!decodeLocal(32768) && !decodeLocal(UINT32_MAX));
    for(unsigned axis=0;axis<3;++axis){LocalVoxel p{};p[axis]=32;CHECK(!localIndex(p));CHECK(!voxelOf({{},p}));CHECK(!relativeCorner({{},p},{}));}
    for(const auto value:{-65,-64,-33,-32,-31,-1,0,1,31,32,33,63,64}) {
        const auto expectedChunk=std::int32_t(std::floor(double(value)/32));
        auto address=addressOf({value,value,value});CHECK(address);
        for(unsigned axis=0;axis<3;++axis)CHECK(address->chunk[axis]==expectedChunk && address->local[axis]==value-expectedChunk*32);
        CHECK((voxelOf(*address)==GlobalVoxel{value,value,value}));
    }
    constexpr auto low=std::int64_t(INT32_MIN)*32,high=std::int64_t(INT32_MAX)*32+31;
    for(auto value:{low,low+31,high-31,high}){auto a=addressOf({value,0,0});CHECK(a && voxelOf(*a)->at(0)==value);}
    CHECK(!addressOf({low-1,0,0}) && !addressOf({high+1,0,0}));
    CHECK(!addressOf({INT64_MIN,0,0}) && !addressOf({INT64_MAX,0,0}));
    CHECK((addressAtRelative({-std::numeric_limits<double>::denorm_min(),0,0})==addressOf({-1,0,0})));
    for(double boundary:{-4.0,-0.125,0.125,4.0}) {
        const auto voxel=std::int64_t(boundary*8);
        CHECK(addressAtRelative({boundary,0,0})==addressOf({voxel,0,0}));
        CHECK(addressAtRelative({std::nextafter(boundary,-INFINITY),0,0})==addressOf({voxel-1,0,0}));
        CHECK(addressAtRelative({std::nextafter(boundary,INFINITY),0,0})==addressOf({voxel,0,0}));
    }
    for(double bad:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::max()})CHECK(!addressAtRelative({bad,0,0}));
    CHECK((addressAtRelative({-std::numeric_limits<double>::denorm_min(),0,0},{INT32_MAX,0,0})==VoxelAddress{{INT32_MAX-1,0,0},{31,0,0}}));
    CHECK(!addressAtRelative({-0.125,0,0},{INT32_MIN,0,0}));
    CHECK(!addressAtRelative({4,0,0},{INT32_MAX,0,0}));
    std::mt19937 random(7142);std::uniform_int_distribution<std::int32_t> chunkValue(INT32_MIN,INT32_MAX);
    for(unsigned i=0;i<4096;++i) {
        VoxelAddress a;Chunk originA,originB;
        for(unsigned axis=0;axis<3;++axis){a.chunk[axis]=chunkValue(random);a.local[axis]=std::uint8_t(random()%32);originA[axis]=chunkValue(random);originB[axis]=chunkValue(random);}
        auto world=voxelOf(a);CHECK(world && addressOf(*world)==a);
        auto first=relativeCorner(a,originA),second=relativeCorner(a,originB);CHECK(first && second);
        CHECK(addressAtRelative(*first,originA)==a && addressAtRelative(*second,originB)==a);
        for(unsigned axis=0;axis<3;++axis)CHECK((*first)[axis]-(*second)[axis]==double(std::int64_t(originB[axis])-originA[axis])*4);
    }
    ResidentChunkLayout basin({-4,0,-4},{8,4,8});CHECK(basin.chunkCount()==256 && basin.voxelCount()==8388608);
    for(std::uint32_t i=0;i<256;++i){auto chunk=basin.decode(i);CHECK(chunk && basin.index(*chunk)==i);}
    CHECK(basin.index({-4,0,-4})==0 && basin.index({3,3,3})==255);
    CHECK(!basin.index({-5,0,0}) && !basin.index({4,0,0}) && !basin.index({0,-1,0}) && !basin.index({0,4,0}) && !basin.decode(256));
    CHECK(rejects([]{ResidentChunkLayout r({}, {0,1,1});}));
    CHECK(rejects([]{ResidentChunkLayout r({}, {65536,65536,1});}));
    CHECK(rejects([]{ResidentChunkLayout r({INT32_MAX,0,0}, {2,1,1});}));
    ResidentChunkLayout edge({INT32_MIN,0,0},{UINT32_MAX,1,1});
    CHECK(edge.index({INT32_MAX-1,0,0})==UINT32_MAX-1 && !edge.index({INT32_MAX,0,0}));
    std::cout<<"Voxel addressing: exhaustive local cells, signed/extreme coordinates, 4096 origin round trips and resident bounds passed\n";
}
