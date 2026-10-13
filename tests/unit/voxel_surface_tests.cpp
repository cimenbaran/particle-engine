#include "core/VoxelSurface.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";return EXIT_FAILURE;}}while(false)
int main() {
    using namespace pw::terrain;using Store=SparseVoxelStore;
    Store store(ResidentChunkLayout({-1,-1,-1},{3,3,3}),65536);
    auto put=[&](VoxelAddress address,std::uint32_t material){return store.setMaterial(address,material,store.revision());};
    auto empty=buildVoxelSurface(store,{});CHECK(empty && empty->faces.empty() && empty->isCurrent(store));
    CHECK(!buildVoxelSurface(store,{2,0,0}) && store.revisionRecordCount()==0);
    const VoxelAddress first{{},{31,10,10}};put(first,65535);
    auto one=buildVoxelSurface(store,{});CHECK(one->faces.size()==6 && !empty->isCurrent(store));
    for(unsigned side=0;side<6;++side) {
        auto face=one->faces[side];CHECK(face.side==side && face.material==65535);
        auto corners=faceCorners(face);CHECK(corners);
        std::array<int,3> a,b,cross;
        for(unsigned axis=0;axis<3;++axis){a[axis]=int((*corners)[1][axis])-(*corners)[0][axis];b[axis]=int((*corners)[2][axis])-(*corners)[0][axis];}
        for(unsigned axis=0;axis<3;++axis)cross[axis]=a[(axis+1)%3]*b[(axis+2)%3]-a[(axis+2)%3]*b[(axis+1)%3];
        for(unsigned axis=0;axis<3;++axis)CHECK(cross[axis]==(axis==side/2?((side&1)?1:-1):0));
        for(auto corner:*corners)CHECK(corner[0]>=31 && corner[0]<=32 && corner[1]>=10 && corner[1]<=11 && corner[2]>=10 && corner[2]<=11);
    }
    CHECK(!faceCorners({32768,0,0}) && !faceCorners({0,6,0}));
    Store other(ResidentChunkLayout({-1,-1,-1},{3,3,3}),65536);
    other.setMaterial(first,65535,0);CHECK(!one->isCurrent(other));
    put({{1,0,0},{0,10,10}},0);CHECK(!one->isCurrent(store));
    auto left=buildVoxelSurface(store,{}),right=buildVoxelSurface(store,{1,0,0});
    CHECK(left->faces.size()==5 && right->faces.size()==5);
    for(auto face:left->faces)CHECK(face.side!=1);
    for(auto face:right->faces)CHECK(face.side!=0);
    put({{1,1,0},{1,1,1}},7);CHECK(left->isCurrent(store) && right->isCurrent(store));
    put({{1,0,0},{0,10,10}},Store::empty);CHECK(!left->isCurrent(store) && !right->isCurrent(store));
    CHECK(buildVoxelSurface(store,{})->faces.size()==6 && buildVoxelSurface(store,{1,0,0})->faces.empty());
    // Fully filled chunk: exactly six 32x32 boundary sheets, no internal faces.
    Store full(ResidentChunkLayout({},{1,1,1}),1);
    for(unsigned i=0;i<voxelsPerChunk;++i)full.setMaterial({{},*decodeLocal(i)},0,full.revision());
    auto shell=buildVoxelSurface(full,{});CHECK(shell->faces.size()==6144);
    std::array<unsigned,6> sides{};for(auto face:shell->faces)++sides[face.side];
    for(auto count:sides)CHECK(count==1024);
    // Independent world-coordinate oracle, including negative chunk seams.
    Store randomStore(ResidentChunkLayout({-1,0,0},{2,1,1}),3);
    std::map<GlobalVoxel,std::uint32_t> cells;std::mt19937 random(18751);
    for(unsigned i=0;i<1600;++i) {
        GlobalVoxel world{std::int64_t(random()%8)-4,std::int64_t(random()%8),std::int64_t(random()%8)};
        auto material=std::uint32_t(random()%3);cells[world]=material;
        randomStore.setMaterial(*addressOf(world),material,randomStore.revision());
    }
    using Key=std::tuple<GlobalVoxel,unsigned,std::uint32_t>;std::set<Key> expected,actual;
    for(auto [world,material]:cells)for(unsigned side=0;side<6;++side) {
        auto neighbor=world;neighbor[side/2]+=(side&1)?1:-1;
        if(!cells.contains(neighbor))expected.emplace(world,side,material);
    }
    for(Chunk chunk:{Chunk{-1,0,0},Chunk{0,0,0}}) {
        auto surface=buildVoxelSurface(randomStore,chunk);
        auto repeat=buildVoxelSurface(randomStore,chunk);CHECK(surface->faces==repeat->faces);
        for(auto face:surface->faces)CHECK(actual.emplace(*voxelOf({chunk,*decodeLocal(face.voxel)}),face.side,face.material).second);
    }
    CHECK(actual==expected);
    for(std::int32_t edge:{INT32_MIN,INT32_MAX}) {
        Store extreme(ResidentChunkLayout({edge,edge,edge},{1,1,1}),1);
        auto local=std::uint8_t(edge==INT32_MIN?0:31);
        extreme.setMaterial({{edge,edge,edge},{local,local,local}},0,0);
        CHECK(buildVoxelSurface(extreme,{edge,edge,edge})->faces.size()==6);
    }
    std::cout<<"Voxel exposed faces, winding, chunk seams, cache revisions and independent oracle passed\n";
}
