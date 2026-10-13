#include "VoxelPublication.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <iostream>
namespace pw {
void runVoxelPublicationTest(const std::filesystem::path& assets) {
    using Store=terrain::SparseVoxelStore;using State=PendingVoxelPublication::State;
    auto require=[](bool ok,const char* text){if(!ok)throw std::runtime_error(text);};
    const auto materials=loadMaterials(assets/"materials/default.json");
    auto store=std::make_unique<Store>(terrain::ResidentChunkLayout({0,0,0},{1,1,1}),std::uint32_t(materials.entries.size()));
    const auto first=*terrain::addressOf({0,0,0}),second=*terrain::addressOf({1,0,0});
    const std::array initial{Store::VoxelEdit{first,0},Store::VoxelEdit{second,1}};store->setMaterials(initial,0);
    VoxelPublicationPtr live=std::make_shared<VoxelPublication>(assets,std::move(store),terrain::Chunk{},materials);auto original=live;
    struct Handles {
        bgfx::DynamicVertexBufferHandle segments=BGFX_INVALID_HANDLE;
        bgfx::ProgramHandle exporter=BGFX_INVALID_HANDLE;bgfx::UniformHandle uniform=BGFX_INVALID_HANDLE;
        bgfx::TextureHandle image=BGFX_INVALID_HANDLE,staging=BGFX_INVALID_HANDLE;
        ~Handles(){if(bgfx::isValid(segments))bgfx::destroy(segments);if(bgfx::isValid(exporter))bgfx::destroy(exporter);if(bgfx::isValid(uniform))bgfx::destroy(uniform);for(auto h:{image,staging})if(bgfx::isValid(h))bgfx::destroy(h);}
    } h;
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    const std::array<float,8> segment{.0625f,.25f,.0625f,.0625f,.0625f,.15625f,.0625f,0};
    h.segments=bgfx::createDynamicVertexBuffer(bgfx::copy(segment.data(),sizeof(segment)),layout,BGFX_BUFFER_COMPUTE_READ);
    h.exporter=loadCompute(assets,"cs_export");h.uniform=bgfx::createUniform("u_sim",bgfx::UniformType::Vec4);
    h.image=bgfx::createTexture2D(4,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE);
    h.staging=bgfx::createTexture2D(4,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    require(bgfx::isValid(h.segments) && bgfx::isValid(h.uniform) && bgfx::isValid(h.image) && bgfx::isValid(h.staging),"Publication test allocation failed");
    GpuVoxelWorldSweep sweep(assets,1);std::uint32_t frame=bgfx::frame();
    auto query=[&](const VoxelPublication& publication) {
        sweep.submit(1,h.segments,publication.occupancy(),publication.source(),publication.origin());
        const float p[]{0,1,0,3};bgfx::setUniform(h.uniform,p);bgfx::setBuffer(0,sweep.output(),bgfx::Access::Read);bgfx::setImage(1,h.image,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(5,h.exporter,1);
        bgfx::TextureRegion src,dst;src.handle=h.image;dst.handle=h.staging;bgfx::blit(7,dst,src);
        std::array<std::uint32_t,16> result{};const auto ready=bgfx::read(dst,result.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();
        return result;
    };
    auto wait=[&](PendingVoxelPublication& edit){for(unsigned i=0;i<64 && edit.state()==State::Waiting;++i){frame=bgfx::frame();SDL_PumpEvents();edit.poll(frame);}require(edit.state()==State::Ready,"Terrain upload did not acknowledge");};
    auto rejects=[&](auto operation){bool caught=false;try{operation();}catch(const std::invalid_argument&){caught=true;}require(caught,"Invalid publication operation accepted");};
    require(original->mesh().faceCount()==10 && query(*original)[7]==1u,"Initial publication collision/mesh mismatch");
    const std::array remove{Store::VoxelEdit{first,Store::empty}};
    PendingVoxelPublication edit(assets,live,remove,1,materials);
    require(edit.state()==State::Prepared && edit.candidate().source().revision()==2 && edit.candidate().mesh().faceCount()==6,"Candidate preparation failed");
    require(live==original && live->source().occupiedVoxelCount()==2 && !edit.publish(live),"Unacknowledged terrain became authoritative");
    rejects([&]{edit.request(UINT16_MAX);});require(edit.state()==State::Prepared,"Invalid view consumed edit");
    edit.request(6);require(edit.state()==State::Waiting && !edit.publish(live),"Pending readback published early");
    wait(edit);require(edit.publish(live) && edit.state()==State::Published && !edit.publish(live),"Terrain publish was not single-use");
    require(live->source().revision()==2 && live->source().occupiedVoxelCount()==1 && live->mesh().faceCount()==6,"Mixed terrain publication");
    auto changed=query(*live);require(changed[7]==0u && changed[10]==2 && changed[11]==0,"New collision retained removed voxel or old revision");
    require(original->source().revision()==1 && original->mesh().faceCount()==10 && query(*original)[7]==1u,"Preparing/publishing mutated retained base");
    rejects([&]{sweep.submit(1,h.segments,original->occupancy(),live->source(),{});});
    rejects([&]{original->mesh().render(10,live->source(),live->occupancy(),{});});
    PendingVoxelPublication competing(assets,original,remove,1,materials);competing.request(6);wait(competing);
    require(!competing.publish(live) && competing.state()==State::Stale && live->source().revision()==2,"Obsolete candidate overwrote newer terrain");
    PendingVoxelPublication staleEarly(assets,original,remove,1,materials);
    VoxelPublicationPtr sameRevision=std::make_shared<VoxelPublication>(assets,original->source().clone(),terrain::Chunk{},materials);
    require(sameRevision->source().revision()==original->source().revision() && !staleEarly.publish(sameRevision) && staleEarly.state()==State::Stale,"Different owner with matching revision accepted");
    rejects([&]{PendingVoxelPublication stale(assets,live,remove,1,materials);});
    const std::array duplicate{remove[0],remove[0]};rejects([&]{PendingVoxelPublication bad(assets,live,duplicate,2,materials);});
    // Water is a valid material ID but cannot build an opaque solid mesh.
    const std::array liquid{Store::VoxelEdit{first,2}};rejects([&]{PendingVoxelPublication bad(assets,live,liquid,2,materials);});
    std::vector<Store::VoxelEdit> hiddenLiquid;
    for(int z=4;z<7;++z)for(int y=4;y<7;++y)for(int x=4;x<7;++x)
        hiddenLiquid.push_back({*terrain::addressOf({x,y,z}),x==5 && y==5 && z==5?2u:0u});
    rejects([&]{PendingVoxelPublication bad(assets,live,hiddenLiquid,2,materials);});
    require(live->source().revision()==2 && query(*live)[7]==0u,"Failed preparation changed current collision");
    PendingVoxelPublication noop(assets,live,remove,2,materials);auto before=live;noop.request(6);wait(noop);
    require(noop.publish(live) && live==before && live->source().revision()==2,"No-op edit manufactured a revision");
    // Destruction must safely drain an in-flight acknowledgement.
    {PendingVoxelPublication abandoned(assets,live,{},2,materials);abandoned.request(6);}
    require(live==before,"Abandoning an upload changed authority");
    std::cout<<"GPU terrain publication passed: staged CPU/mesh/collision, upload acknowledgement, old snapshot retention, stale competing edits, failed preparation, no-op and in-flight teardown\n";
}
}
