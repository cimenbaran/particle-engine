#include "core/BasinWorld.hpp"
#include <iostream>
#define CHECK(v) do{if(!(v)){std::cerr<<"FAIL "<<__LINE__<<": " #v "\n";return 1;}}while(false)
int main() {
    using namespace pw::terrain;
    auto first=buildBasin(1337,4,0,1,3),same=buildBasin(1337,4,0,1,3),different=buildBasin(42,4,0,1,3);
    CHECK(first->layout().chunkCount()==256 && first->revision()==1);
    CHECK(first->layout().minimum()==(Chunk{-4,0,-4}));
    auto at=[&](int x,int y,int z){return *first->materialAt(*addressOf({x,y,z}));};
    CHECK(at(0,3,0)==0 && at(0,4,0)==SparseVoxelStore::empty);
    CHECK(at(-128,15,0)==0 && at(-128,16,0)==SparseVoxelStore::empty);
    CHECK(at(-80,31,-32)==0 && at(-80,32,-32)==SparseVoxelStore::empty);
    CHECK(at(-24,4,-32)==1 && at(-24,3,-32)==0);
    CHECK(at(40,0,40)==0 && at(40,1,40)==SparseVoxelStore::empty);
    CHECK(at(40,4,-40)==3 && at(40,5,-40)==SparseVoxelStore::empty);
    std::array<std::uint32_t,voxelsPerChunk> a,b,c;std::uint64_t counted=0,differences=0;
    for(unsigned slot=0;slot<256;++slot) {
        const auto chunk=*first->layout().decode(slot);first->copyChunk(chunk,a);same->copyChunk(chunk,b);different->copyChunk(chunk,c);
        CHECK(a==b);
        for(unsigned index=0;index<voxelsPerChunk;++index) {
            if(a[index]!=SparseVoxelStore::empty){++counted;CHECK(a[index]==0 || a[index]==1 || a[index]==3);}
            if(a[index]!=c[index]){++differences;CHECK(a[index]==1 || c[index]==1);}
        }
    }
    CHECK(counted==first->occupiedVoxelCount() && counted>300000 && differences>500);
    bool rejected=false;try{buildBasin(0,2,0,1,2);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    std::cout<<"Basin generation passed: "<<counted<<" voxels, "<<differences<<" seed differences, full resident occupancy checked\n";
}
