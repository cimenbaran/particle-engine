#include "GpuVoxelWorld.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
namespace pw {
namespace {
template<class T>struct WorldHandle {
    T value;explicit WorldHandle(T handle):value(handle){if(!bgfx::isValid(handle))throw std::runtime_error("World test allocation failed");}
    ~WorldHandle(){bgfx::destroy(value);}WorldHandle(const WorldHandle&)=delete;
};
struct WorldSegment {std::array<float,4> start,end;};
void requireWorld(bool valid,const char* message){if(!valid)throw std::runtime_error(message);}
}
void runGpuVoxelWorldTest(const std::filesystem::path& assets) {
    using namespace terrain;
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    WorldHandle exporter(loadCompute(assets,"cs_export"));
    WorldHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned batches=0,queries=0,hits=0,overlaps=0,misses=0,outside=0,invalid=0;
    auto check=[&](const SparseVoxelStore& store,const GpuVoxelWorld& world,Chunk origin,const std::vector<WorldSegment>& segments,GpuVoxelWorldSweep* reuse=nullptr) {
        const auto count=std::uint32_t(segments.size());requireWorld(count>0 && count<=4096,"Invalid world test size");
        struct Occupied {std::uint32_t address,material;reference::VoxelVector lower;};
        std::vector<Occupied> occupied;
        std::array<std::uint32_t,voxelsPerChunk> chunk;
        for(unsigned slot=0;slot<store.layout().chunkCount();++slot) {
            const auto coord=*store.layout().decode(slot);store.copyChunk(coord,chunk);
            for(unsigned local=0;local<voxelsPerChunk;++local)if(chunk[local]!=SparseVoxelStore::empty)
                occupied.push_back({slot*voxelsPerChunk+local,chunk[local],*relativeCorner({coord,*decodeLocal(local)},origin)});
        }
        auto owned=reuse?nullptr:std::make_unique<GpuVoxelWorldSweep>(assets,count);auto& query=reuse?*reuse:*owned;
        WorldHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(segments.data(),count*sizeof(WorldSegment)),lanes,BGFX_BUFFER_COMPUTE_READ));
        query.submit(0,input.value,world,store,origin);
        WorldHandle image(bgfx::createTexture2D(3,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        WorldHandle staging(bgfx::createTexture2D(3,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        WorldHandle inputImage(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        WorldHandle inputStaging(bgfx::createTexture2D(2,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        auto exportBuffer=[&](bgfx::DynamicVertexBufferHandle buffer,bgfx::TextureHandle texture,float stride) {
            const float p[]{0,float(count),0,stride};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,buffer,bgfx::Access::Read);
            bgfx::setImage(1,texture,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        };
        exportBuffer(query.output(),image.value,3);exportBuffer(input.value,inputImage.value,2);
        std::vector<std::array<float,12>> actual(count);std::vector<WorldSegment> original(count);
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=inputImage.value;dst.handle=inputStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
        while(bgfx::frame()<ready)SDL_PumpEvents();
        requireWorld(std::memcmp(original.data(),segments.data(),count*sizeof(WorldSegment))==0,"World query mutated segments");
        const auto lower=*relativeCorner({store.layout().minimum(),{}},origin);const auto dims=store.layout().dimensions();
        for(unsigned i=0;i<count;++i) {
            const auto& row=actual[i];const auto& segment=segments[i];
            auto bits=[&](unsigned lane){return std::bit_cast<std::uint32_t>(row[lane]);};
            requireWorld(bits(10)==std::uint32_t(world.revision()) && bits(11)==std::uint32_t(world.revision()>>32),"World query revision mismatch");
            for(unsigned lane=0;lane<7;++lane)requireWorld(std::isfinite(row[lane]),"World query nonfinite output");
            bool valid=segment.start[3]==.0625f && segment.end[3]==0,inside=true;
            reference::VoxelVector start,end;
            for(unsigned axis=0;axis<3;++axis) {
                start[axis]=segment.start[axis];end[axis]=segment.end[axis];
                valid&=std::isfinite(start[axis]) && std::isfinite(end[axis]) && std::abs(start[axis])<=64 && std::abs(end[axis])<=64 && std::abs(segment.end[axis]-segment.start[axis])<=.125f;
                inside&=std::min(start[axis],end[axis])-.0625>=lower[axis] && std::max(start[axis],end[axis])+.0625<=lower[axis]+dims[axis]*4;
            }
            unsigned kind=!valid?3:!inside?4:0;std::uint32_t material=UINT32_MAX,address=UINT32_MAX;
            reference::VoxelSweepResult expected;expected.centre=valid?end:reference::VoxelVector{};
            if(valid && inside)for(const auto& voxel:occupied) {
                auto candidate=reference::sweepVoxelSphere(start,end,voxel.lower);
                if(!candidate.hit)continue;
                const auto candidateKind=candidate.initialOverlap?2u:1u;
                if(kind==0 || candidate.time<expected.time || (candidate.time==expected.time &&
                   ((candidateKind==2 && kind!=2) || (candidateKind==kind && voxel.address<address)))) {
                    kind=candidateKind;material=voxel.material;address=voxel.address;expected=candidate;
                }
            }
            auto fail=[&](const char* what){throw std::runtime_error(std::string(what)+" batch "+std::to_string(batches)+" slot "+std::to_string(i)+" kind "+std::to_string(bits(7))+" expected "+std::to_string(kind)+" material "+std::to_string(bits(8))+" address "+std::to_string(bits(9))+" expected address "+std::to_string(address));};
            if(bits(7)!=kind)fail("World classification mismatch");
            if(bits(8)!=material || bits(9)!=address)fail("World selected voxel mismatch");
            if(std::abs(row[3]-expected.time)>2e-5)fail("World contact time mismatch");
            for(unsigned axis=0;axis<3;++axis) {
                if(std::abs(row[axis]-expected.normal[axis])>5e-4)fail("World normal mismatch");
                if(std::abs(row[axis+4]-expected.centre[axis])>2e-5)fail("World centre mismatch");
            }
            if(kind==0)++misses;else if(kind==1)++hits;else if(kind==2)++overlaps;else if(kind==3)++invalid;else ++outside;
        }
        ++batches;queries+=count;
    };
    SparseVoxelStore store(ResidentChunkLayout({-4,0,-4},{8,4,8}),65536);
    std::vector<WorldSegment> segments;
    auto add=[&](std::array<float,3> start,std::array<float,3> end){segments.push_back({{start[0],start[1],start[2],.0625f},{end[0],end[1],end[2],0}});};
    // Adjacent voxels across all chunk axes, including negative seams. Include
    // material zero and the largest material ID, both distinct from empty.
    for(int z:{-33,-32,-1,0,31,32})for(int y:{31,32,63,64})for(int x:{-33,-32,-1,0,31,32}) {
        const auto address=*addressOf({x,y,z});store.setMaterial(address,(x+y+z)%2?65535:0,store.revision());
        const float px=x*.125f,py=y*.125f,pz=z*.125f;
        add({px-.125f,py+.0625f,pz+.0625f},{px,py+.0625f,pz+.0625f});
        add({px+.0625f,py+.0625f,pz+.0625f},{px+.0625f,py+.0625f,pz+.0625f});
    }
    // Exact shared-face tangency must include the voxel on either side.
    add({-.0625f,4.0625f,.0625f},{-.0625f,4.0625f,.0625f});
    add({.0625f,3.8125f,.0625f},{.0625f,3.9375f,.0625f});
    add({15.9375f,8,0},{15.9375f,8,0});
    add({std::nextafter(15.9375f,16.0f),8,0},{std::nextafter(15.9375f,16.0f),8,0});
    add({0,.0625f,0},{0,.0625f,0});add({0,0,0},{0,0,0});
    add({0,8,0},{.25f,8,0});add({0,8,0},{0,8,0});segments.back().start[0]=std::numeric_limits<float>::quiet_NaN();
    for(unsigned axis=0;axis<3;++axis)for(bool upper:{false,true}) {
        const float bound=axis==1?(upper?15.9375f:.0625f):(upper?15.9375f:-15.9375f);
        std::array<float,3> position{0,8,0};position[axis]=bound;add(position,position);
        position[axis]=std::nextafter(bound,upper?64.0f:-64.0f);add(position,position);
    }
    add({0,8,0},{0,8,0});segments.back().start[3]=0;
    add({0,8,0},{0,8,0});segments.back().end[3]=1;
    add({0,8,0},{0,8,0});segments.back().end[2]=std::numeric_limits<float>::infinity();
    std::mt19937 random(482019);
    while(segments.size()<2049) {
        const float x=(int(random()%80)-40)*.125f,y=(24+random()%48)*.125f,z=(int(random()%80)-40)*.125f;
        const auto delta=[&]{return (int(random()%2001)-1000)/8000.0f;};
        std::array<float,3> start{x+delta(),y+delta(),z+delta()},end=start;
        for(auto& value:end)value+=delta()*.99f;
        add(start,end);
    }
    GpuVoxelWorld first(store,{});
    requireWorld(first.payloadBytes()==33554496 && first.isCurrent(store,{}),"Basin snapshot payload/current mismatch");
    GpuVoxelWorldSweep reusable(assets,std::uint32_t(segments.size()));
    check(store,first,{},segments,&reusable);
    auto rejected=[&](auto action){bool threw=false;try{action();}catch(const std::invalid_argument&){threw=true;}requireWorld(threw,"Invalid world operation was accepted");};
    GpuVoxelWorldSweep query(assets,1);
    WorldHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(segments.data(),sizeof(WorldSegment)),lanes,BGFX_BUFFER_COMPUTE_READ));
    SparseVoxelStore other(store.layout(),65536);
    rejected([&]{query.submit(0,input.value,first,other,{});});
    rejected([&]{query.submit(0,input.value,first,store,{1,0,0});});
    rejected([&]{query.submit(0,query.output(),first,store,{});});
    rejected([&]{query.submit(0,first.buffer(),first,store,{});});
    const auto changed=*addressOf({-33,31,-33});
    store.setMaterial(changed,*store.materialAt(changed),store.revision());
    requireWorld(first.isCurrent(store,{}),"No-op invalidated world snapshot");
    const std::array edits{SparseVoxelStore::VoxelEdit{changed,SparseVoxelStore::empty},
        SparseVoxelStore::VoxelEdit{*addressOf({32,64,32}),29}};
    store.setMaterials(edits,store.revision());
    requireWorld(!first.isCurrent(store,{}),"Terrain deletion left stale snapshot current");
    rejected([&]{query.submit(0,input.value,first,store,{});});
    GpuVoxelWorld deleted(store,{});check(store,deleted,{},segments,&reusable);
    store.setMaterial(changed,17,store.revision());
    GpuVoxelWorld recreated(store,{});check(store,recreated,{},segments,&reusable);
    // Same occupancy under a new shared origin; integer subtraction must precede
    // float conversion. Old origin cannot be used with the rebased snapshot.
    const Chunk rebasedOrigin{1,1,-1};GpuVoxelWorld rebased(store,rebasedOrigin);
    auto moved=segments;
    for(auto& segment:moved)for(unsigned axis=0;axis<3;++axis){segment.start[axis]-=rebasedOrigin[axis]*4;segment.end[axis]-=rebasedOrigin[axis]*4;}
    check(store,rebased,rebasedOrigin,moved,&reusable);
    rejected([&]{query.submit(0,input.value,rebased,store,{});});
    for(auto extreme:{INT32_MIN,INT32_MAX}) {
        const Chunk origin{extreme,extreme,extreme};SparseVoxelStore far(ResidentChunkLayout(origin,{1,1,1}),1);
        far.setMaterial({origin,{15,15,15}},0,0);GpuVoxelWorld snapshot(far,origin);
        check(far,snapshot,origin,{{{1.75f,1.9375f,1.9375f,.0625f},{1.875f,1.9375f,1.9375f,0}},{{2,2,2,.0625f},{2,2,2,0}}});
    }
    SparseVoxelStore empty(ResidentChunkLayout({0,0,0},{1,1,1}),1);GpuVoxelWorld emptyWorld(empty,{});
    check(empty,emptyWorld,{},{{{1,1,1,.0625f},{1.125f,1,1,0}}});
    GpuVoxelWorldSweep zero(assets,0);zero.submit(0,input.value,emptyWorld,empty,{});bgfx::frame();
    rejected([&]{GpuVoxelWorld tooFar(store,{100,0,0});});
    SparseVoxelStore tooLarge(ResidentChunkLayout({0,0,0},{257,1,1}),1);
    rejected([&]{GpuVoxelWorld excessive(tooLarge,{});});
    requireWorld(hits && overlaps && misses && outside && invalid,"World query classification coverage missing");
    std::cout<<"GPU resident voxel sweep passed: "<<batches<<" batches, "<<queries<<" queries, "<<hits<<" contacts, "<<overlaps<<" overlaps, "<<misses<<" misses; full basin, seams, edits, stale rejection, origin rebasing and immutable input\n";
}
}
