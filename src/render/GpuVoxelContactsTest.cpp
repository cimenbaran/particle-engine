#include "GpuJacobiContacts.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
namespace pw {
namespace {
template<class T>struct ConstraintHandle {T value;explicit ConstraintHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Voxel constraint allocation failed");}~ConstraintHandle(){bgfx::destroy(value);}ConstraintHandle(const ConstraintHandle&)=delete;};
using Vector=reference::VoxelVector;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
terrain::VoxelAddress address(Vector p) {
    terrain::VoxelAddress result;
    for(unsigned a=0;a<3;++a){const auto cell=std::int64_t(std::llround(p[a]*8));auto chunk=cell/32,local=cell%32;if(local<0){local+=32;--chunk;}result.chunk[a]=std::int32_t(chunk);result.local[a]=std::uint8_t(local);}
    return result;
}
std::vector<ParticleGpu> pair(Vector p,Vector correction) {
    std::vector<ParticleGpu> result(2);const double magnitude=std::hypot(correction[0],correction[1],correction[2]);
    for(unsigned i=0;i<2;++i){auto& q=result[i];q.positionRadius[3]=.0625f;q.velocityInvMass[3]=i?0:1;q.thermalState[0]=293.15f;q.thermalState[3]=17;q.identity[1]=1;q.identity[2]=i;q.identity[3]=8;for(unsigned a=0;a<3;++a)q.positionRadius[a]=float(p[a]-(i?correction[a]*(.125-magnitude)/magnitude:0));}
    return result;
}
Vector unconstrained(const std::vector<ParticleGpu>& p) {
    Vector delta;for(unsigned a=0;a<3;++a)delta[a]=double(p[0].positionRadius[a])-p[1].positionRadius[a];
    const double distance=std::hypot(delta[0],delta[1],delta[2]);
    for(unsigned a=0;a<3;++a)delta[a]=p[0].positionRadius[a]+delta[a]*std::min(.0625,std::max(0.0,.125-distance))/distance;
    return delta;
}
Vector singleReference(const std::vector<ParticleGpu>& p,Vector lower) {
    Vector start;for(unsigned a=0;a<3;++a)start[a]=p[0].positionRadius[a];auto end=unconstrained(p);
    auto hit=reference::sweepVoxelSphere(start,end,lower);
    if(!hit.hit)return end;
    require(!hit.initialOverlap,"Constraint fixture starts overlapping");
    double inward=0;for(unsigned a=0;a<3;++a)inward+=(end[a]-start[a])*hit.normal[a];
    if(inward>=0)return end;
    for(unsigned a=0;a<3;++a)end[a]=hit.centre[a]+2e-6*hit.normal[a]+(end[a]-start[a]-hit.normal[a]*inward)*(1-hit.time);
    return end;
}
}
void runGpuVoxelContactsTest(const std::filesystem::path& assets) {
    using namespace terrain;
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    ConstraintHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    ConstraintHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    std::array<MaterialGpu,2> materials{};materials[1].identity[0]=1;materials[1].identity[1]=1;
    ConstraintHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    unsigned cases=0;
    auto check=[&](std::vector<Vector> occupied,std::vector<ParticleGpu> input,Vector expected,unsigned error=0,unsigned iterations=1,bool badBaseline=false,unsigned terrainMaterial=0,const std::vector<Vector>& expectedPositions={},GranularGridProfile profile=GranularGridProfile::Basin) {
        SparseVoxelStore store(ResidentChunkLayout({-4,0,-4},{8,4,8}),2);
        for(auto p:occupied)store.setMaterial(address(p),terrainMaterial,store.revision());
        GpuVoxelWorld world(store,{});VoxelContactContext context{world,store,{}};
        GpuJacobiContacts solver(assets,std::uint32_t(input.size()),1,.0625f,true,{},profile);
        auto baseline=input;if(badBaseline)for(unsigned a=0;a<3;++a)baseline[0].positionRadius[a]=float(occupied[0][a]+.0625);
        ConstraintHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),std::uint32_t(input.size()*sizeof(ParticleGpu))),layout,BGFX_BUFFER_COMPUTE_READ));
        ConstraintHandle old(bgfx::createDynamicVertexBuffer(bgfx::copy(baseline.data(),std::uint32_t(baseline.size()*sizeof(ParticleGpu))),layout,BGFX_BUFFER_COMPUTE_READ));
        const auto count=std::uint32_t(input.size());
        auto current=state.value;
        for(unsigned iteration=0;iteration<iterations;++iteration){solver.submit(0,current,materialBuffer.value,2,old.value,{},&context);current=solver.output();}
        ConstraintHandle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ConstraintHandle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ConstraintHandle sourceImage(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ConstraintHandle sourceStaging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ConstraintHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ConstraintHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        for(auto pair:{std::pair{current,image.value},std::pair{state.value,sourceImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,solver.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(count),original(count);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=sourceImage.value;dst.handle=sourceStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
        src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,stats.data()));
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if(stats[0]!=error)throw std::runtime_error("Voxel Jacobi status case "+std::to_string(cases)+": "+std::to_string(stats[0])+" expected "+std::to_string(error));
        require(!std::memcmp(input.data(),original.data(),count*sizeof(ParticleGpu)),"Voxel Jacobi mutated source");
        if(error)require(!std::memcmp(input.data(),actual.data(),count*sizeof(ParticleGpu)),"Voxel Jacobi failed whole-iteration rollback");
        else {
            for(unsigned i=0;i<count;++i) {
                require(!std::memcmp(input[i].velocityInvMass,actual[i].velocityInvMass,48),"Voxel Jacobi modified velocity/thermal/identity");
                require(input[i].positionRadius[3]==actual[i].positionRadius[3],"Voxel Jacobi changed radius");
                if(!input[i].velocityInvMass[3] || !(input[i].identity[1]&1) || input[i].identity[0]==1)require(!std::memcmp(&input[i],&actual[i],sizeof(ParticleGpu)),"Voxel Jacobi changed excluded/static slot");
                else {
                    Vector p;double displacement=0;for(unsigned a=0;a<3;++a){p[a]=actual[i].positionRadius[a];const double d=p[a]-input[i].positionRadius[a];displacement+=d*d;}
                    require(std::sqrt(displacement)<=iterations*.0625+2e-6,"Voxel Jacobi correction exceeded limit");
                    for(auto lower:occupied)require(reference::voxelDistance(p,lower)>=.0625-1e-8,"Voxel Jacobi stored penetrating endpoint");
                    if(!expectedPositions.empty())for(unsigned a=0;a<3;++a)
                        require(std::abs(p[a]-expectedPositions[i/2][a])<=8e-6,"Voxel Jacobi random corner CPU mismatch");
                }
            }
            if(iterations==8 && count==2 && input[1].velocityInvMass[3]>0) {
                double squared=0;for(unsigned a=0;a<3;++a){const double d=double(actual[0].positionRadius[a])-actual[1].positionRadius[a];squared+=d*d;}
                require(std::sqrt(squared)>=.125-.0003,"Voxel Jacobi failed to separate dynamic pair against wall");
            }
            if(iterations==1)for(unsigned a=0;a<3;++a)if(std::abs(actual[0].positionRadius[a]-expected[a])>8e-6)
                throw std::runtime_error("Voxel Jacobi CPU/analytic position case "+std::to_string(cases)+" axis "+std::to_string(a)+": "+std::to_string(actual[0].positionRadius[a])+" expected "+std::to_string(expected[a]));
        }
        // Host revision, owner and origin rejection must precede any output change.
        const auto previous=solver.output();
        auto reject=[&](const VoxelContactContext& invalid){bool caught=false;try{solver.submit(0,state.value,materialBuffer.value,2,old.value,{},&invalid);}catch(const std::invalid_argument&){caught=true;}require(caught && solver.output().idx==previous.idx,"Voxel Jacobi accepted stale context");};
        VoxelContactContext shifted{world,store,{1,0,0}};reject(shifted);
        SparseVoxelStore foreign(ResidentChunkLayout({-4,0,-4},{8,4,8}),2);VoxelContactContext wrong{world,foreign,{}};reject(wrong);
        store.setMaterial(address({15,15,15}),0,store.revision());reject(context);
        ++cases;
    };
    for(double x:{-12.0,0.0,11.875})for(unsigned axis=0;axis<3;++axis)for(double sign:{-1.0,1.0}) {
        Vector lower{x,2,0},p{x+.0625,2.0625,.0625},delta{};
        p[axis]=lower[axis]+(sign>0?.125:0)+sign*(.0625+.005);
        delta[axis]=-sign*.025;delta[(axis+1)%3]=.0125;
        auto input=pair(p,delta);check({lower},input,singleReference(input,lower));
    }
    // Rounded edge/corner, tangent escape and a second perpendicular surface.
    for(unsigned axes:{2u,3u}) {
        Vector lower{0,2,0},p{.0625,2.0625,.0625},delta{};
        for(unsigned a=0;a<axes;++a){p[a]=lower[a]-(.0625+.005)/std::sqrt(double(axes));delta[a]=.018;}
        if(axes==2)delta[2]=.008;
        auto input=pair(p,delta);check({lower},input,singleReference(input,lower));
    }
    auto wedge=pair({-.0675,2.0575,.0625},{.02,.02,.01});
    check({{0,2,0},{-.125,2.125,0}},wedge,{-.062502,2.062498,.0725});
    check({{0,2,0},{-.125,2.125,0}},wedge,{},0,8);
    auto empty=pair({-11.5,2,.0625},{.02,.01,0});check({},empty,unconstrained(empty));
    auto surface=pair({-.0625,2.0625,.0625},{0,.02,0});check({{0,2,0}},surface,unconstrained(surface));
    auto dynamic=pair({-.0625,2.0625,.0625},{.04,0,0});dynamic[1].velocityInvMass[3]=1;
    check({{0,2,0}},dynamic,{},0,8);
    auto excluded=surface;auto inert=surface[0];inert.positionRadius[0]=.0625f;inert.identity[0]=1;inert.identity[2]=2;excluded.push_back(inert);
    inert.identity[0]=0;inert.identity[1]=0;inert.identity[2]=3;excluded.push_back(inert);
    check({{0,2,0}},excluded,unconstrained(surface));
    auto overlap=pair({-.06,2.0625,.0625},{-.02,0,0});check({{0,2,0}},overlap,{},16);
    auto valid=pair({-.0675,2.0625,.0625},{.02,.01,0});
    check({{0,2,0}},valid,singleReference(valid,{0,2,0}),0,1,false,0,{},GranularGridProfile::Legacy);
    check({{0,2,0}},valid,{},32,1,true);
    check({{0,2,0}},valid,{},4,1,false,1);
    // One bad dynamic slot rolls back an otherwise valid moving pair too.
    auto mixed=valid;auto other=pair({-11.5,2,0},{.02,0,0});mixed.insert(mixed.end(),other.begin(),other.end());
    for(unsigned i=0;i<mixed.size();++i)mixed[i].identity[2]=i;
    mixed[0].positionRadius[0]=-.06f;check({{0,2,0}},mixed,{},16);
    std::mt19937 random(91337);std::uniform_real_distribution<double> component(.1,1);
    std::vector<Vector> voxels,expectedBatch;std::vector<ParticleGpu> batch;
    for(unsigned i=0;i<256;++i) {
        Vector lower{-12.0+double(i%16),2,-7.0+double(i/16)},n,t,p,delta;
        for(unsigned a=0;a<3;++a)n[a]=component(random)*((i&(1u<<a))?1:-1);
        const double magnitude=std::hypot(n[0],n[1],n[2]);for(auto& value:n)value/=magnitude;
        t={-n[1],n[0],0};const double tangent=std::hypot(t[0],t[1]);for(auto& value:t)value/=tangent;
        for(unsigned a=0;a<3;++a){p[a]=lower[a]+(n[a]>0?.125:0)+n[a]*.0645;delta[a]=-n[a]*.03+t[a]*.012;}
        auto bodies=pair(p,delta);expectedBatch.push_back(singleReference(bodies,lower));voxels.push_back(lower);
        for(auto body:bodies){body.identity[2]=std::uint32_t(batch.size());batch.push_back(body);}
    }
    check(voxels,batch,expectedBatch[0],0,1,false,0,expectedBatch);
    std::cout<<"GPU voxel-constrained Jacobi passed: "<<cases<<" fixtures; CPU/analytic faces, edges, corners, multi-surface slide, repeated iterations, immutable state, rollback and revision gates\n";
}
}
