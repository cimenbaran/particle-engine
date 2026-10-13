#include "GpuVoxelMotion.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/BasinWorld.hpp"
#include <bit>
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
namespace pw {
namespace {
template<class T>struct MotionHandle {T value;explicit MotionHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Motion test allocation failed");}~MotionHandle(){bgfx::destroy(value);}MotionHandle(const MotionHandle&)=delete;};
using Vector=reference::VoxelVector;
struct ReferenceResult {Vector p,v;unsigned events=0,normals=0;};
ReferenceResult referenceMotion(const ParticleGpu& particle,const terrain::SparseVoxelStore& store,const std::vector<terrain::VoxelAddress>& occupied,
                                const std::array<MaterialGpu,3>& materials,float step,float cap,float threshold,unsigned budget) {
    ReferenceResult r;for(unsigned a=0;a<3;++a){r.p[a]=particle.positionRadius[a];r.v[a]=particle.velocityInvMass[a];}
    const double speed=std::hypot(r.v[0],r.v[1],r.v[2]);if(speed>cap)for(double& v:r.v)v*=cap/speed;
    const auto& own=materials[particle.identity[0]];double remaining=step;
    Vector lo=*terrain::relativeCorner({store.layout().minimum(),{}},{}),hi=lo;
    for(unsigned a=0;a<3;++a){hi[a]+=store.layout().dimensions()[a]*4-.0625;lo[a]+=.0625;}
    while(remaining>0) {
        Vector times;for(unsigned a=0;a<3;++a)times[a]=r.v[a]==0?2:std::clamp(((r.v[a]>0?hi[a]:lo[a])-r.p[a])/r.v[a],0.0,2.0);
        const double wall=std::min({times[0],times[1],times[2]}),duration=std::min(wall,remaining);
        Vector end;for(unsigned a=0;a<3;++a)end[a]=std::clamp(r.p[a]+r.v[a]*duration,lo[a],hi[a]);
        reference::VoxelSweepResult hit;std::uint32_t selected=UINT32_MAX,material=UINT32_MAX;
        for(auto address:occupied) {
            auto candidate=reference::sweepVoxelSphere(r.p,end,*terrain::relativeCorner(address,{}));
            if(!candidate.hit)continue;
            if(candidate.initialOverlap)throw std::runtime_error("CPU motion initial overlap");
            double entering=0;for(unsigned a=0;a<3;++a)entering+=(end[a]-r.p[a])*candidate.normal[a];
            if(entering>=0)continue;
            auto index=*store.layout().index(address.chunk)*terrain::voxelsPerChunk+*terrain::localIndex(address.local);
            if(!hit.hit || candidate.time<hit.time || (candidate.time==hit.time && index<selected)){hit=candidate;selected=index;material=*store.materialAt(address);}
        }
        if(!hit.hit && wall>remaining){r.p=end;break;}
        if(r.events++>=budget)throw std::runtime_error("CPU motion exhausted events");
        if(hit.hit) {
            const auto& other=materials[material];double incoming=0;
            for(unsigned a=0;a<3;++a)incoming+=r.v[a]*hit.normal[a];
            const double outgoing=-incoming>threshold?-std::min(own.mechanical[2],other.mechanical[2])*incoming:0;
            Vector tangent;for(unsigned a=0;a<3;++a)tangent[a]=r.v[a]-hit.normal[a]*incoming;
            const double speed=std::hypot(tangent[0],tangent[1],tangent[2]);
            const double scale=speed>0?std::max(0.0,1-std::sqrt(double(own.mechanical[1])*other.mechanical[1])*(outgoing-incoming)/speed):0;
            for(unsigned a=0;a<3;++a){r.v[a]=hit.normal[a]*outgoing+tangent[a]*scale;r.p[a]=hit.centre[a]+hit.normal[a]*2e-6;}
            remaining=std::max(0.0,remaining-duration*hit.time);++r.normals;
        } else {
            const auto old=r.v;r.p=end;remaining=std::max(0.0,remaining-wall);Vector tangent{},change{};
            for(unsigned a=0;a<3;++a){if(times[a]==wall){r.v[a]=std::abs(old[a])>threshold?-own.mechanical[2]*old[a]:0;++r.normals;}else tangent[a]=r.v[a];change[a]=r.v[a]-old[a];}
            const double speed=std::hypot(tangent[0],tangent[1],tangent[2]),support=std::hypot(change[0],change[1],change[2]);
            if(speed>0)for(unsigned a=0;a<3;++a)if(times[a]!=wall)r.v[a]*=std::max(0.0,1-own.mechanical[1]*support/speed);
        }
    }
    return r;
}
}
void runGpuVoxelMotionTest(const std::filesystem::path& assets) {
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    using namespace terrain;
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    MotionHandle exporter(loadCompute(assets,"cs_export")),statExporter(loadCompute(assets,"cs_pool_stats"));
    MotionHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    std::array<MaterialGpu,3> materials{};for(unsigned i=0;i<3;++i){materials[i].identity[0]=i;materials[i].mechanical[1]=.4f;materials[i].mechanical[2]=.6f;}
    materials[1].mechanical[1]=.9f;materials[1].mechanical[2]=.2f;materials[2].identity[1]=1;
    unsigned cases=0,particles=0;
    auto body=[](Vector p,Vector v,unsigned material=0) {ParticleGpu result{};for(unsigned a=0;a<3;++a){result.positionRadius[a]=float(p[a]);result.velocityInvMass[a]=float(v[a]);}result.positionRadius[3]=.0625f;result.velocityInvMass[3]=1;result.thermalState[0]=293.15f;result.identity[0]=material;result.identity[1]=1;result.identity[3]=1;return result;};
    auto check=[&](const SparseVoxelStore& store,const std::vector<VoxelAddress>& occupied,std::vector<ParticleGpu> input,
                   float dt=.01f,unsigned expectedError=0,unsigned upstream=0,unsigned budget=16,GpuVoxelMotion* reuse=nullptr,bool wedge=false,bool basinReplay=false) {
        for(unsigned i=0;i<input.size();++i)input[i].identity[2]=i;
        const auto count=std::uint32_t(input.size()),storage=std::max(1u,count);auto upload=input;if(upload.empty())upload.resize(1);
        GpuVoxelWorld world(store,{});auto owned=reuse?nullptr:std::make_unique<GpuVoxelMotion>(assets,count);auto& motion=reuse?*reuse:*owned;
        MotionHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(upload.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        MotionHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
        const std::array<std::uint32_t,4> upstreamData{upstream,0,0,0};
        MotionHandle status(bgfx::createDynamicIndexBuffer(bgfx::copy(upstreamData.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        motion.submit(0,state.value,materialBuffer.value,3,status.value,world,store,{},dt,12,.5f,budget);
        MotionHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        MotionHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        MotionHandle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        MotionHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        MotionHandle statsImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        MotionHandle statsStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count)for(auto pair:{std::pair{motion.output(),image.value},std::pair{state.value,oldImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,motion.status(),bgfx::Access::Read);bgfx::setImage(1,statsImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statExporter.value,1);
        std::vector<ParticleGpu> actual(storage),old(storage);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=statsImage.value;dst.handle=statsStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count){src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,old.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if(stats[0]!=expectedError)throw std::runtime_error("Voxel motion status case "+std::to_string(cases)+": "+std::to_string(stats[0])+" expected "+std::to_string(expectedError));
        unsigned capped=0,events=0,normals=0;
        for(unsigned i=0;i<count;++i) {
            require(std::memcmp(&old[i],&input[i],64)==0,"Voxel motion mutated input");const auto& source=input[i];const auto& result=actual[i];
            if(expectedError || !(source.identity[1]&1) || source.identity[0]==2 || source.velocityInvMass[3]==0){require(std::memcmp(&source,&result,64)==0,"Voxel rollback/excluded particle changed");continue;}
            ReferenceResult expected;
            if(basinReplay) {
                expected.p={result.positionRadius[0],result.positionRadius[1],result.positionRadius[2]};expected.v={result.velocityInvMass[0],result.velocityInvMass[1],result.velocityInvMass[2]};
                if(source.velocityInvMass[0]==0 && source.velocityInvMass[2]==0)
                    require(std::hypot(result.velocityInvMass[0],result.velocityInvMass[1],result.velocityInvMass[2])<1e-5,"Four-corner shell failed to constrain resting grain");
                if(source.velocityInvMass[0]>.8f)require(result.positionRadius[0]-source.positionRadius[0]>.003f,"Captured basin skin case lost free X travel");
            } else if(wedge) {
                // Opposing rounded edges constrain XY, leaving the Z direction
                // free. A zero-all-velocity workaround must fail this fixture.
                require(std::abs(result.velocityInvMass[0])<5e-4 && std::abs(result.velocityInvMass[1])<5e-4 && std::abs(result.velocityInvMass[2]-1)<1e-5,"Contact cone removed free tangential motion");
                require(result.positionRadius[1]>=3.875f && result.positionRadius[1]<3.876f && std::abs(result.positionRadius[0]+5.6875f)<2e-6,"Contact cone failed narrow-gap pose");
                expected.p={result.positionRadius[0],result.positionRadius[1],source.positionRadius[2]+double(dt)};expected.v={0,0,1};
            } else expected=referenceMotion(source,store,occupied,materials,dt,12,.5f,budget);
            events+=expected.events;normals+=expected.normals;
            const double before=std::hypot(double(source.velocityInvMass[0]),source.velocityInvMass[1],source.velocityInvMass[2]);if(before>12)++capped;
            const double after=std::hypot(double(result.velocityInvMass[0]),result.velocityInvMass[1],result.velocityInvMass[2]);require(after<=std::min(before,12.0)+2e-5,"Voxel response gained energy");
            for(unsigned a=0;a<3;++a)if(std::abs(result.positionRadius[a]-expected.p[a])>1e-5 || std::abs(result.velocityInvMass[a]-expected.v[a])>5e-4)
                throw std::runtime_error("Voxel motion CPU disagreement case "+std::to_string(cases)+" slot "+std::to_string(i)+" axis "+std::to_string(a)+" p "+std::to_string(result.positionRadius[a])+" expected "+std::to_string(expected.p[a])+" v "+std::to_string(result.velocityInvMass[a])+" expected "+std::to_string(expected.v[a])+" events "+std::to_string(stats[2]));
            require(result.positionRadius[3]==source.positionRadius[3] && result.velocityInvMass[3]==source.velocityInvMass[3] && std::memcmp(result.thermalState,source.thermalState,32)==0,"Voxel motion changed identity/thermal/mass");
            Vector position{result.positionRadius[0],result.positionRadius[1],result.positionRadius[2]};
            for(auto address:occupied)require(reference::voxelDistance(position,*relativeCorner(address,{}))>=.0625-1e-7,"Voxel motion published penetration");
        }
        // A complete shell can resolve both wedge normals in one collision event.
        if(wedge)require(stats[2]>=1 && stats[2]<=16 && stats[3]==stats[2],"Narrow gap contact event accounting failed");
        if(!expectedError && !wedge && !basinReplay)require(stats[1]==capped && stats[2]==events && stats[3]==normals,"Voxel motion counters disagree");
        ++cases;particles+=count;
    };
    SparseVoxelStore voxel(ResidentChunkLayout({-1,-1,-1},{2,2,2}),3);const auto single=*addressOf({0,0,0});voxel.setMaterial(single,1,0);
    std::vector<ParticleGpu> faces;
    for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
        Vector p{.0625,.0625,.0625},v{};p[axis]=sign<0?-.1:.225;v[axis]=-sign*8;faces.push_back(body(p,v));
    }
    faces.push_back(body({-.0625,.0625,.0625},{-4,0,0})); // outward contact
    faces.push_back(body({-.0625,.0625,.0625},{0,2,0})); // tangent contact
    faces.push_back(body({-.1,-.03,.0625},{10,0,0})); // rounded edge
    faces.push_back(body({-.1,-.02,-.02},{10,0,0})); // rounded corner
    faces.push_back(body({1,1,1},{1000,0,0})); // cap
    auto excluded=body({.0625,.0625,.0625},{1,2,3});excluded.velocityInvMass[3]=0;faces.push_back(excluded);
    excluded.velocityInvMass[3]=1;excluded.identity[1]=0;faces.push_back(excluded);
    excluded.identity[1]=1;excluded.identity[0]=2;faces.push_back(excluded);
    check(voxel,{single},faces);
    for(float restitution:{0.0f,1.0f})for(float friction:{0.0f,1.0f}) {
        materials[0].mechanical[1]=materials[1].mechanical[1]=friction;materials[0].mechanical[2]=materials[1].mechanical[2]=restitution;
        check(voxel,{single},faces);
    }
    materials[0].mechanical[1]=.4f;materials[1].mechanical[1]=.9f;materials[0].mechanical[2]=.6f;materials[1].mechanical[2]=.2f;
    std::vector<ParticleGpu> randomBodies;std::mt19937 random(193441);
    for(unsigned i=0;i<257;++i){double y=(int(random()%301)-150)/1000.0,z=(int(random()%301)-150)/1000.0;randomBodies.push_back(body({-.15,y,z},{8+double(random()%401)/100,0,0}));}
    for(unsigned count:{1u,127u,128u,129u,255u,256u,257u})check(voxel,{single},{randomBodies.begin(),randomBodies.begin()+count});
    GpuVoxelMotion reused(assets,257);check(voxel,{single},randomBodies,.01f,0,0,16,&reused);check(voxel,{single},randomBodies,.005f,0,0,16,&reused);
    SparseVoxelStore empty(ResidentChunkLayout({-1,-1,-1},{2,2,2}),3);
    check(empty,{}, {body({3.9,1,1},{12,0,0}),body({3.9,3.9,3.9},{8,8,8}),body({-3.9,1,1},{-8,0,0})});
    SparseVoxelStore corner(ResidentChunkLayout({-1,-1,-1},{2,2,2}),3);std::vector<VoxelAddress> cornerCells;
    for(int z=-2;z<3;++z)for(int a=0;a<4;++a)for(auto global:{GlobalVoxel{-1,a,z},GlobalVoxel{a,-1,z}}){auto address=*addressOf(global);cornerCells.push_back(address);corner.setMaterial(address,1,corner.revision());}
    materials[0].mechanical[1]=materials[1].mechanical[1]=0;
    check(corner,cornerCells,{body({.09,.11,.0625},{-8,-8,0})});
    check(corner,cornerCells,{body({.09,.11,.0625},{-8,-8,0})},.01f,4,0,1);
    SparseVoxelStore wedgeWorld(ResidentChunkLayout({-2,0,-2},{1,2,1}),3);std::vector<VoxelAddress> wedgeCells;
    for(int z=-60;z< -56;++z)for(int x:{-47,-45}){auto address=*addressOf({x,30,z});wedgeCells.push_back(address);wedgeWorld.setMaterial(address,1,wedgeWorld.revision());}
    materials[0].mechanical[2]=materials[1].mechanical[2]=0;
    check(wedgeWorld,wedgeCells,{body({std::nextafter(-5.6875f,0.0f),3.8755f,-7.1875f},{0,-1,1})},.01f,0,0,16,nullptr,true);
    // Exact prediction inputs captured from the dense interacting basin.
    const std::array<std::array<std::uint32_t,16>,3> basinCases{{
        {3225596409u,1075316310u,3230793729u,1031798784u,1063122435u,3173475230u,0u,1058829479u,1133679411u,0u,0u,1068883462u,1u,1u,95u,1u},
        {3227779071u,1076104597u,3231187343u,1031798784u,952768685u,3204461930u,3171002964u,1058829479u,1133679411u,0u,0u,1070491300u,1u,1u,86u,1u},
        {3230793726u,1078986427u,3232104447u,1031798784u,0u,3173475230u,0u,1058829479u,1133679411u,0u,0u,1072448668u,1u,1u,3u,1u}
    }};
    auto basin=buildBasin(1337,3,0,1,2);materials[0].mechanical[1]=.7f;materials[0].mechanical[2]=.18f;materials[1].mechanical[1]=.5f;materials[1].mechanical[2]=.12f;
    for(auto words:basinCases) {
        auto particle=std::bit_cast<ParticleGpu>(words);std::vector<VoxelAddress> local;
        GlobalVoxel centre{};for(unsigned a=0;a<3;++a)centre[a]=std::int64_t(std::floor(particle.positionRadius[a]*8));
        for(int z=-4;z<=4;++z)for(int y=-4;y<=4;++y)for(int x=-4;x<=4;++x){auto a=*addressOf({centre[0]+x,centre[1]+y,centre[2]+z});auto m=basin->materialAt(a);if(m && *m!=SparseVoxelStore::empty)local.push_back(a);}
        check(*basin,local,{particle},1.0f/240,0,0,16,nullptr,false,true);
    }
    auto bad=body({.0625,.0625,.0625},{1,0,0});check(voxel,{single},{faces.front(),bad},.01f,2);
    bad=body({1,1,1},{1,0,0});bad.positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(voxel,{single},{faces.front(),bad},.01f,2);
    check(voxel,{single},{faces.front()},.01f,1,8);check(voxel,{single},{},.01f);
    std::cout<<"GPU voxel motion passed: "<<cases<<" cases, "<<particles<<" particles; CPU response, energy, containment, materials, walls, seams, counters and whole-pass rollback\n";
}
}
