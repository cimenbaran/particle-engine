#include "core/SparseVoxelStore.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
template<class F> bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    using namespace pw::terrain;using Store=SparseVoxelStore;using Edit=Store::Edit;
    ResidentChunkLayout region({-1,-1,-1},{3,3,3});Store store(region,65536);
    CHECK(rejects([&]{Store invalid(region,0);}));CHECK(rejects([&]{Store invalid(region,65537);}));
    const VoxelAddress a{{0,0,0},{0,31,0}};
    CHECK(store.materialAt(a)==Store::empty && store.revisionRecordCount()==0);
    CHECK(!store.materialAt({{2,0,0},{}}) && !store.materialAt({{},{32,0,0}}));
    CHECK(store.setMaterial(a,Store::empty,0)==Edit::Unchanged && store.revisionRecordCount()==0);
    CHECK(store.setMaterial(a,65535,0)==Edit::Changed);
    CHECK(store.materialAt(a)==65535 && store.occupiedVoxelCount()==1 && store.allocatedChunkCount()==1);
    CHECK(store.occupancyPayloadBytes()==131072 && store.revisionRecordCount()==4);
    CHECK((store.revisions({0,0,0})==Store::Revisions{1,1}));
    for(Chunk neighbor:{Chunk{-1,0,0},Chunk{0,1,0},Chunk{0,0,-1}})CHECK((store.revisions(neighbor)==Store::Revisions{0,1}));
    CHECK((store.revisions({-1,1,0})==Store::Revisions{}));
    CHECK(store.setMaterial(a,0,0)==Edit::Stale && store.materialAt(a)==65535);
    CHECK(store.setMaterial(a,65535,1)==Edit::Unchanged && store.revision()==1);
    CHECK(rejects([&]{store.setMaterial(a,65536,1);}));
    CHECK(rejects([&]{store.setMaterial({{2,0,0},{}},0,1);}));
    CHECK(rejects([&]{store.setMaterial({{},{32,0,0}},0,1);}));
    CHECK(store.revision()==1 && store.occupiedVoxelCount()==1);
    std::array<std::uint32_t,voxelsPerChunk> copy{};
    CHECK(store.copyChunk(a.chunk,copy)==store.revisions(a.chunk));
    CHECK(copy[*localIndex(a.local)]==65535 && copy[0]==Store::empty);
    const VoxelAddress interior{{},{12,13,14}};
    CHECK(store.setMaterial(interior,0,1)==Edit::Changed);
    CHECK((store.revisions({-1,0,0})==Store::Revisions{0,1}));
    CHECK(store.setMaterial(a,Store::empty,2)==Edit::Changed);
    CHECK(store.setMaterial(interior,Store::empty,3)==Edit::Changed);
    CHECK(store.occupiedVoxelCount()==0 && store.allocatedChunkCount()==0 && store.revisionRecordCount()==4);
    auto deleted=store.revisions({});CHECK(deleted->content==4);
    CHECK(store.setMaterial(interior,0,4)==Edit::Changed && store.revisions({})->content>deleted->content);
    CHECK(store.copyChunk({1,1,1},copy)==Store::Revisions{});
    CHECK(copy[0]==Store::empty && copy.back()==Store::empty);
    copy.fill(123);CHECK(!store.copyChunk({2,0,0},copy) && copy[0]==123 && copy.back()==123);
    CHECK(store.revisionRecordCount()==4 && store.allocatedChunkCount()==1);
    for(unsigned axis=0;axis<3;++axis)for(std::uint8_t face:{std::uint8_t(0),std::uint8_t(31)}) {
        Store boundary(region,1);VoxelAddress cell{{},{15,15,15}};cell.local[axis]=face;
        CHECK(boundary.setMaterial(cell,0,0)==Edit::Changed);
        Chunk neighbor{};neighbor[axis]=face==0?-1:1;
        CHECK((boundary.revisions(neighbor)==Store::Revisions{0,1}));
        CHECK(boundary.revisionRecordCount()==2 && boundary.allocatedChunkCount()==1);
        CHECK(boundary.setMaterial(cell,Store::empty,1)==Edit::Changed);
        CHECK((boundary.revisions(neighbor)==Store::Revisions{0,2}));
        CHECK(boundary.allocatedChunkCount()==0 && boundary.revisionRecordCount()==2);
    }
    // Independent sparse world oracle; repeated replacement, removal and stale writes.
    Store randomStore(region,7);std::map<GlobalVoxel,std::uint32_t> oracle;
    std::mt19937 random(71823);std::uint64_t revision=0;
    for(unsigned i=0;i<3000;++i) {
        GlobalVoxel world;for(auto& axis:world)axis=std::int64_t(random()%96)-32;
        if(i%3==0 && !oracle.empty())world=oracle.begin()->first;
        auto address=addressOf(world);CHECK(address);
        const auto material=i%4==0?Store::empty:random()%7;
        auto prior=oracle.find(world);const auto old=prior==oracle.end()?Store::empty:prior->second;
        if(revision)CHECK(randomStore.setMaterial(*address,material,revision-1)==Edit::Stale);
        CHECK(randomStore.setMaterial(*address,material,revision)==(old==material?Edit::Unchanged:Edit::Changed));
        if(old!=material)++revision;
        if(material==Store::empty)oracle.erase(world);else oracle[world]=material;
        CHECK(randomStore.materialAt(*address)==material && randomStore.revision()==revision);
        CHECK(randomStore.occupiedVoxelCount()==oracle.size());
    }
    std::uint64_t counted=0;
    for(unsigned slot=0;slot<region.chunkCount();++slot) {
        auto chunk=*region.decode(slot);CHECK(randomStore.copyChunk(chunk,copy));
        for(unsigned local=0;local<voxelsPerChunk;++local) {
            auto world=*voxelOf({chunk,*decodeLocal(local)});auto expected=oracle.find(world);
            CHECK(copy[local]==(expected==oracle.end()?Store::empty:expected->second));
            counted+=copy[local]!=Store::empty;
        }
    }
    CHECK(counted==oracle.size());
    for(std::int32_t extreme:{INT32_MIN,INT32_MAX}) {
        Store edge(ResidentChunkLayout({extreme,extreme,extreme},{1,1,1}),1);
        for(std::uint8_t local:{std::uint8_t(0),std::uint8_t(31)}) {
            CHECK(edge.setMaterial({{extreme,extreme,extreme},{local,local,local}},0,edge.revision())==Edit::Changed);
            CHECK(edge.revisionRecordCount()==1);
        }
    }
    std::cout<<"Sparse voxel materials, revisions, neighbor invalidation and independent world oracle passed\n";
}
