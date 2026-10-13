#include "VoxelTerrain.hpp"
#include <SDL3/SDL.h>
#include <bx/math.h>
#include <cmath>
#include <iostream>
namespace pw {
void runVoxelTerrainTest(const std::filesystem::path& assets) {
    auto require=[](bool v,const char* m){if(!v)throw std::runtime_error(m);};
    MaterialLibrary materials;MaterialDesc red;red.gpu.color[0]=1;red.gpu.color[3]=1;materials.entries.push_back(red);
    terrain::SparseVoxelStore store(terrain::ResidentChunkLayout({-1,-1,-1},{2,2,2}),1);
    const auto address=*terrain::addressOf({-1,0,0});store.setMaterial(address,0,0);
    GpuVoxelWorld world(store,{});VoxelTerrain mesh(assets,store,world,{},materials);
    require(mesh.faceCount()==6 && mesh.payloadBytes()==528,"Single voxel mesh payload mismatch");
    struct Targets {
        bgfx::TextureHandle color=BGFX_INVALID_HANDLE,depth=BGFX_INVALID_HANDLE,readback=BGFX_INVALID_HANDLE;
        bgfx::FrameBufferHandle frame=BGFX_INVALID_HANDLE;
        ~Targets(){bgfx::setViewFrameBuffer(10,BGFX_INVALID_HANDLE);bgfx::setViewClear(10,BGFX_CLEAR_NONE);if(bgfx::isValid(frame))bgfx::destroy(frame);for(auto h:{color,depth,readback})if(bgfx::isValid(h))bgfx::destroy(h);}
    } targets;
    targets.color=bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_RT);
    targets.depth=bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::D24S8,BGFX_TEXTURE_RT);
    targets.readback=bgfx::createTexture2D(64,64,false,1,bgfx::TextureFormat::RGBA8,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
    for(auto h:{targets.color,targets.depth,targets.readback})require(bgfx::isValid(h),"Terrain render test target failed");
    const bgfx::TextureHandle attachments[]{targets.color,targets.depth};targets.frame=bgfx::createFrameBuffer(2,attachments,false);
    require(bgfx::isValid(targets.frame),"Terrain framebuffer failed");
    auto draw=[&](const VoxelTerrain& terrain,const GpuVoxelWorld& snapshot,unsigned side,bool empty) {
        bx::Vec3 centre{-.0625f,.0625f,.0625f},eye=centre,up{0,1,0};
        float* coordinate=side/2==0?&eye.x:side/2==1?&eye.y:&eye.z;*coordinate+=(side&1)?1:-1;
        if(side/2==1)up={0,0,1};
        float view[16],projection[16];bx::mtxLookAt(view,eye,centre,up);
        bx::mtxOrtho(projection,-.125f,.125f,-.125f,.125f,.1f,10,0,bgfx::getCaps()->homogeneousDepth);
        bgfx::setViewRect(10,0,0,64,64);bgfx::setViewFrameBuffer(10,targets.frame);bgfx::setViewTransform(10,view,projection);
        bgfx::setViewClear(10,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,0x000000ff,1.0f,0);bgfx::touch(10);terrain.render(10,store,snapshot,{});
        bgfx::TextureRegion src,dst;src.handle=targets.color;dst.handle=targets.readback;bgfx::blit(11,dst,src);
        std::array<std::uint8_t,64*64*4> pixels{};const auto ready=bgfx::read(dst,pixels.data());while(bgfx::frame()<ready)SDL_PumpEvents();
        constexpr double light[3]{-.39036003,.78072006,.48795004};
        const int expected=empty?0:int(std::lround(255*(.3+.7*std::max(0.0,light[side/2]*((side&1)?1:-1)))));
        for(unsigned y=20;y<44;++y)for(unsigned x=20;x<44;++x) {
            const unsigned offset=4*(y*64+x);
            require(std::abs(int(pixels[offset])-expected)<=2 && pixels[offset+1]==0 && pixels[offset+2]==0,"Voxel face color/depth image mismatch");
        }
        require(pixels[(2*64+2)*4]==0,"Voxel render exceeded expected bounds");
    };
    for(unsigned side=0;side<6;++side)draw(mesh,world,side,false);
    auto rejected=[&](auto action){bool failed=false;try{action();}catch(const std::invalid_argument&){failed=true;}require(failed,"Stale terrain rendering accepted");};
    rejected([&]{mesh.render(10,store,world,{1,0,0});});
    store.setMaterial(address,terrain::SparseVoxelStore::empty,store.revision());
    rejected([&]{mesh.render(10,store,world,{});});
    rejected([&]{VoxelTerrain stale(assets,store,world,{},materials);});
    GpuVoxelWorld deleted(store,{});rejected([&]{mesh.render(10,store,deleted,{});});
    VoxelTerrain empty(assets,store,deleted,{},materials);require(empty.faceCount()==0 && empty.payloadBytes()==0,"Empty terrain mesh not empty");draw(empty,deleted,0,true);
    store.setMaterial(address,0,store.revision());store.setMaterial(*terrain::addressOf({0,0,0}),0,store.revision());
    GpuVoxelWorld seamWorld(store,{});VoxelTerrain seam(assets,store,seamWorld,{},materials);
    require(seam.faceCount()==10,"Terrain renderer retained hidden chunk seam faces");
    std::cout<<"GPU voxel terrain renderer passed: six face images, material color/depth, negative coordinates, chunk seam, empty rebuild and stale publication rejection\n";
}
}
