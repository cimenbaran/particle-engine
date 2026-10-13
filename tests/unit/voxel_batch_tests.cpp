#include "core/VoxelSurface.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#include <new>
// This executable alone replaces allocation to exercise every preparation
// failure point. Injection is disabled before assertions or fixture teardown.
static std::int64_t allocationsBeforeFailure=-1;
void* operator new(std::size_t size) {
    if(allocationsBeforeFailure==0)throw std::bad_alloc();
    if(allocationsBeforeFailure>0)--allocationsBeforeFailure;
    if(auto p=std::malloc(size?size:1))return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
template<class F> bool rejects(F f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main() {
    using namespace pw::terrain;using Store=SparseVoxelStore;using Edit=Store::Edit;using Write=Store::VoxelEdit;
    const ResidentChunkLayout region({-1,-1,-1},{3,3,3});Store store(region,3);
    const VoxelAddress left{{-1,0,0},{31,7,8}},right{{0,0,0},{0,7,8}},inside{{0,0,0},{10,10,10}};
    const std::array first{Write{left,0},Write{right,1},Write{inside,2}};
    CHECK(store.setMaterials(first,0)==Edit::Changed && store.revision()==1);
    CHECK(store.occupiedVoxelCount()==3 && store.allocatedChunkCount()==2);
    CHECK((store.revisions(left.chunk)==Store::Revisions{1,1}));
    CHECK((store.revisions(right.chunk)==Store::Revisions{1,1}));
    auto oldLeft=buildVoxelSurface(store,left.chunk),oldRight=buildVoxelSurface(store,right.chunk);
    const auto records=store.revisionRecordCount();
    CHECK(store.setMaterials(first,1)==Edit::Unchanged && store.revision()==1 && store.revisionRecordCount()==records);
    CHECK(store.setMaterials({},1)==Edit::Unchanged && store.setMaterials({},0)==Edit::Stale);
    const std::array change{Write{right,Store::empty},Write{left,2},Write{inside,0}};
    CHECK(store.setMaterials(change,0)==Edit::Stale && store.materialAt(right)==1);
    // Invalid trailing data must not commit an earlier valid removal.
    for(const Write bad:{Write{{{4,0,0},{}},0},Write{{{},{32,0,0}},0},Write{inside,3}}) {
        const std::array invalid{Write{left,Store::empty},bad};
        CHECK(rejects([&]{store.setMaterials(invalid,1);}));
        CHECK(store.materialAt(left)==0 && store.revision()==1 && oldLeft->isCurrent(store));
    }
    for(auto material:{0u,Store::empty}) {
        const std::array duplicate{Write{left,Store::empty},Write{left,material}};
        CHECK(rejects([&]{store.setMaterials(duplicate,1);}));
    }
    CHECK(store.setMaterials(change,1)==Edit::Changed && store.revision()==2);
    CHECK(store.occupiedVoxelCount()==2 && store.allocatedChunkCount()==2);
    CHECK(store.materialAt(left)==2 && store.materialAt(right)==Store::empty && store.materialAt(inside)==0);
    CHECK(!oldLeft->isCurrent(store) && !oldRight->isCurrent(store));
    CHECK((store.revisions(left.chunk)==Store::Revisions{2,2}));
    const std::array remove{Write{left,Store::empty},Write{inside,Store::empty}};
    CHECK(store.setMaterials(remove,2)==Edit::Changed && store.occupiedVoxelCount()==0 && store.allocatedChunkCount()==0);
    CHECK(store.revisionRecordCount()==records);
    CHECK(store.setMaterials(first,3)==Edit::Changed && store.revisions(left.chunk)->content==4);
    // A no-op boundary destination must not invalidate its untouched neighbor.
    Store noop(region,3);noop.setMaterial(left,0,0);auto cached=buildVoxelSurface(noop,right.chunk);
    const std::array mix{Write{left,0},Write{{{-1,0,0},{5,5,5}},1}};
    CHECK(noop.setMaterials(mix,1)==Edit::Changed && cached->isCurrent(noop));
    // Same logical batch in reverse order has identical contents and revisions.
    Store reversed(region,3);auto backward=first;std::reverse(backward.begin(),backward.end());
    CHECK(reversed.setMaterials(backward,0)==Edit::Changed);
    Store forward(region,3);CHECK(forward.setMaterials(first,0)==Edit::Changed);
    for(unsigned i=0;i<region.chunkCount();++i)CHECK(forward.revisions(*region.decode(i))==reversed.revisions(*region.decode(i)));
    unsigned failures=0;bool completed=false,sawMetadataResidue=false;
    for(std::int64_t failAt=0;failAt<64;++failAt) {
        Store interrupted(region,3);interrupted.setMaterials(first,0);
        const auto beforeRecords=interrupted.revisionRecordCount();
        const VoxelAddress added{{0,1,1},{0,0,0}};
        const std::array transaction{Write{left,Store::empty},Write{inside,1},Write{added,2}};
        allocationsBeforeFailure=failAt;bool failed=false;
        try {interrupted.setMaterials(transaction,1);}catch(const std::bad_alloc&){failed=true;}
        allocationsBeforeFailure=-1;
        if(failed) {
            ++failures;sawMetadataResidue|=interrupted.revisionRecordCount()>beforeRecords;
            CHECK(interrupted.revision()==1 && interrupted.occupiedVoxelCount()==3 && interrupted.allocatedChunkCount()==2);
            CHECK(interrupted.materialAt(left)==0 && interrupted.materialAt(right)==1 && interrupted.materialAt(inside)==2);
            CHECK(interrupted.materialAt(added)==Store::empty);
            for(unsigned slot=0;slot<region.chunkCount();++slot) {
                auto chunk=*region.decode(slot);CHECK(interrupted.revisions(chunk)==forward.revisions(chunk));
            }
        } else {
            completed=true;CHECK(interrupted.revision()==2 && interrupted.occupiedVoxelCount()==3);
            CHECK(interrupted.materialAt(left)==Store::empty && interrupted.materialAt(inside)==1 && interrupted.materialAt(added)==2);
            break;
        }
    }
    CHECK(completed && failures>=6 && sawMetadataResidue);
    // Independent map oracle over mixed batches; global revision advances once
    // for each batch with any changed destination, regardless of edit count.
    Store randomStore(region,3);std::map<GlobalVoxel,std::uint32_t> oracle;std::mt19937 random(41236);std::uint64_t revision=0;
    for(unsigned batch=0;batch<400;++batch) {
        std::map<GlobalVoxel,std::uint32_t> proposal;
        for(unsigned i=0;i<30;++i) {
            GlobalVoxel voxel;for(auto& v:voxel)v=std::int64_t(random()%8)-4;
            proposal[voxel]=random()%4==0?Store::empty:random()%3;
        }
        std::vector<Write> writes;bool changed=false;
        for(auto [voxel,material]:proposal) {
            auto prior=oracle.find(voxel);changed|=(prior==oracle.end()?Store::empty:prior->second)!=material;
            writes.push_back({*addressOf(voxel),material});
        }
        if(revision)CHECK(randomStore.setMaterials(writes,revision-1)==Edit::Stale);
        CHECK(randomStore.setMaterials(writes,revision)==(changed?Edit::Changed:Edit::Unchanged));
        revision+=changed;
        for(auto [voxel,material]:proposal){if(material==Store::empty)oracle.erase(voxel);else oracle[voxel]=material;}
        CHECK(randomStore.revision()==revision && randomStore.occupiedVoxelCount()==oracle.size());
        for(std::int64_t z=-4;z<4;++z)for(std::int64_t y=-4;y<4;++y)for(std::int64_t x=-4;x<4;++x) {
            GlobalVoxel voxel{x,y,z};auto expected=oracle.find(voxel);
            CHECK(randomStore.materialAt(*addressOf(voxel))==(expected==oracle.end()?Store::empty:expected->second));
        }
    }
    for(std::int32_t edge:{INT32_MIN,INT32_MAX}) {
        Store extreme(ResidentChunkLayout({edge,edge,edge},{1,1,1}),1);
        const std::array corners{Write{{{edge,edge,edge},{0,0,0}},0},Write{{{edge,edge,edge},{31,31,31}},0}};
        CHECK(extreme.setMaterials(corners,0)==Edit::Changed && extreme.occupiedVoxelCount()==2);
        CHECK(extreme.revision()==1 && extreme.revisionRecordCount()==1);
    }
    std::cout<<"Allocation failure points exercised: "<<failures<<"\n";
    std::cout<<"Atomic voxel batches, revisions, cache invalidation and independent oracle passed\n";
}
