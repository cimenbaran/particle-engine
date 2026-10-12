#include "../reference/SpatialIndex.hpp"
#include <cstdlib>
#include <bit>
#include <iostream>
#include <limits>
#include <random>
#define CHECK(x) do {if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}} while(false)
int main() {
    using namespace pw::reference;
    pw::SpatialGrid grid({0,0,0},0.125f,{-2,-2,-2},{4,4,4});
    const auto empty=buildSpatialIndex(grid,{});
    CHECK(empty.records.empty() && empty.ranges.empty() && empty.find(0).empty());
    const float nan=std::numeric_limits<float>::quiet_NaN();
    std::vector<SpatialPoint> points{
        {90,{0.13f,0.13f,0.13f}}, {7,{-0.01f,-0.01f,-0.01f}},
        {2,{-0.125f,-0.125f,-0.125f}}, {40,{-0.25f,-0.25f,-0.25f}},
        {1,{nan,nan,nan},false,true}, {5,{10,10,10},true,false}};
    const auto original=points;
    auto index=buildSpatialIndex(grid,points);
    const std::vector<CellSlot> expected{{0,40},{21,2},{21,7},{63,90}};
    const std::vector<CellRange> expectedRanges{{0,0,1},{21,1,3},{63,3,4}};
    CHECK(index.records==expected && index.ranges==expectedRanges);
    for(std::size_t i=0;i<points.size();++i) {
        CHECK(points[i].slot==original[i].slot && points[i].active==original[i].active && points[i].physical==original[i].physical);
        // Inactive NaNs are allowed and must remain bit-identical; NaN != NaN.
        for(unsigned axis=0;axis<3;++axis)
            CHECK(std::bit_cast<std::uint32_t>(points[i].position[axis])==std::bit_cast<std::uint32_t>(original[i].position[axis]));
    }
    CHECK(index.find(21).size()==2 && index.find(21)[0].slot==2);
    CHECK(index.find(20).empty() && index.find(22).empty() && index.find(UINT32_MAX).empty());
    // Shuffle input repeatedly: key and membership results are independent of
    // emission order, and persistent slot numbers are preserved verbatim.
    std::mt19937 random(42);
    for(int iteration=0;iteration<20;++iteration) {
        std::shuffle(points.begin(),points.end(),random);
        auto result=buildSpatialIndex(grid,points);
        CHECK(result.records==expected && result.ranges==expectedRanges);
    }
    for(unsigned count:{1u,127u,128u,129u,255u,256u,257u,1024u}) {
        points.clear();
        for(unsigned i=0;i<count;++i) points.push_back({count-i,{0,0,0}});
        auto result=buildSpatialIndex(grid,points);
        CHECK(result.ranges.size()==1 && result.records.size()==count);
        CHECK(result.ranges[0].key==42 && result.ranges[0].begin==0 && result.ranges[0].end==count);
        for(unsigned i=0;i<count;++i) CHECK(result.records[i].slot==i+1 && result.records[i].key==42);
    }
    // Fill each cell once in deliberately reversed input order. Expected keys
    // are known independently from x + 4*y + 16*z offsets.
    points.clear();
    for(int z=3;z>=0;--z) for(int y=3;y>=0;--y) for(int x=3;x>=0;--x) {
        const auto key=std::uint32_t(x+4*y+16*z);
        points.push_back({1000-key,{(x-2)*0.125f,(y-2)*0.125f,(z-2)*0.125f}});
    }
    index=buildSpatialIndex(grid,points);
    CHECK(index.records.size()==64 && index.ranges.size()==64);
    for(std::uint32_t key=0;key<64;++key) {
        CHECK(index.find(key).size()==1 && index.find(key)[0].slot==1000-key);
        CHECK(index.ranges[key].begin==key && index.ranges[key].end==key+1);
    }
    auto rejects=[&](std::vector<SpatialPoint> bad) {
        try {buildSpatialIndex(grid,bad);return false;} catch(const std::invalid_argument&) {return true;}
    };
    CHECK(rejects({{1,{0,0,0}},{1,{0,0,0}}}));
    CHECK(rejects({{UINT32_MAX,{0,0,0}}}));
    CHECK(rejects({{1,{0,0,0}},{2,{0.25f,0,0}}}));
    CHECK(rejects({{1,{nan,0,0}}}));
    CHECK(rejects({{1,{0,0,0},false,true},{1,{0,0,0},true,false}}));
    std::cout<<"CPU spatial sort, occupied ranges, exact lookup and slot preservation passed\n";
}
