#include "GpuVoxelMotion.hpp"
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
#include <limits>
#include <random>
namespace pw {
namespace {
template<class T>struct FinishHandle {T value;explicit FinishHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Voxel finish allocation failed");}~FinishHandle(){bgfx::destroy(value);}FinishHandle(const FinishHandle&)=delete;};
using Vector=reference::VoxelVector;
double dot(Vector a,Vector b){double result=0;for(unsigned i=0;i<3;++i)result+=a[i]*b[i];return result;}
struct Contact {Vector normal;double friction;};
std::vector<Contact> contacts(Vector p,const std::vector<Vector>& voxels,float own,float other) {
    std::vector<Contact> result;const Vector lo{-15.9375,.0625,-15.9375},hi{15.9375,15.9375,15.9375};
    for(unsigned a=0;a<3;++a){Vector n{};if(p[a]<=lo[a]+4e-6)n[a]=1;else if(p[a]>=hi[a]-4e-6)n[a]=-1;if(n[a])result.push_back({n,own});}
    for(auto lower:voxels){Vector n;for(unsigned a=0;a<3;++a)n[a]=p[a]-std::clamp(p[a],lower[a],lower[a]+.125);const double distance=std::sqrt(dot(n,n));if(distance>.062504)continue;for(auto& a:n)a/=distance;result.push_back({n,std::sqrt(double(own)*other)});}
    return result;
}
// Independent double active-set solve. Solve Gram equations for multipliers,
// then form the projected velocity from normal impulses (GPU uses cross axes).
Vector referenceFinish(Vector v,const std::vector<Contact>& c,bool* capped=nullptr) {
    const auto cap=[&](Vector q){const double speed=std::hypot(q[0],q[1],q[2]);if(capped)*capped=speed>12;if(speed>12)for(auto& a:q)a*=12*.9999995231628418/speed;return q;};
    const auto feasible=[&](Vector q){for(auto a:c)if(dot(q,a.normal)<-1e-9)return false;return true;};
    if(feasible(v))return cap(v);
    Vector best{};double retained=0,budget=0;
    auto consider=[&](unsigned i,unsigned j,double a,double b){if(a<0 || b<0)return;Vector q=v;for(unsigned k=0;k<3;++k)q[k]+=a*c[i].normal[k]+b*c[j].normal[k];const double speed=std::hypot(q[0],q[1],q[2]);if(speed>retained && feasible(q)){best=q;retained=speed;budget=a*c[i].friction+b*c[j].friction;}};
    for(unsigned i=0;i<c.size();++i){const double di=dot(v,c[i].normal);consider(i,i,-di,0);for(unsigned j=0;j<i;++j){const double cosine=dot(c[i].normal,c[j].normal),denominator=1-cosine*cosine;if(denominator<=1e-12)continue;const double dj=dot(v,c[j].normal);consider(i,j,(-di+cosine*dj)/denominator,(-dj+cosine*di)/denominator);}}
    const double length=std::sqrt(dot(best,best));if(length>0)for(auto& a:best)a*=std::max(0.0,1-budget/length);return cap(best);
}
terrain::VoxelAddress address(Vector lower) {
    terrain::VoxelAddress result;for(unsigned a=0;a<3;++a){const auto cell=std::int64_t(std::llround(lower[a]*8));auto chunk=cell/32,local=cell%32;if(local<0){local+=32;--chunk;}result.chunk[a]=std::int32_t(chunk);result.local[a]=std::uint8_t(local);}return result;
}
}
void runGpuVoxelFinishTest(const std::filesystem::path& assets) {
    using namespace terrain;
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    FinishHandle exporter(loadCompute(assets,"cs_export")),statExporter(loadCompute(assets,"cs_pool_stats"));
    FinishHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned cases=0,particles=0;
    auto body=[](Vector p,Vector v){ParticleGpu q{};for(unsigned a=0;a<3;++a){q.positionRadius[a]=float(p[a]);q.velocityInvMass[a]=float(v[a]);}q.positionRadius[3]=.0625f;q.velocityInvMass[3]=1;q.thermalState[0]=293.15f;q.thermalState[3]=13;q.identity[1]=1;q.identity[3]=7;return q;};
    auto check=[&](std::vector<Vector> occupied,std::vector<ParticleGpu> input,float own=.4f,float terrainFriction=.9f,unsigned expectedError=0,unsigned upstream=0,bool repeat=false,unsigned terrainMaterial=1) {
        SparseVoxelStore store(ResidentChunkLayout({-4,0,-4},{8,4,8}),3);for(auto lower:occupied)store.setMaterial(address(lower),terrainMaterial,store.revision());
        std::array<MaterialGpu,3> materials{};for(unsigned i=0;i<3;++i){materials[i].identity[0]=i;materials[i].mechanical[1]=i==1?terrainFriction:own;}materials[2].identity[1]=1;
        for(unsigned i=0;i<input.size();++i)input[i].identity[2]=i;
        const auto count=std::uint32_t(input.size()),storage=std::max(1u,count);auto upload=input;if(upload.empty())upload.resize(1);
        GpuVoxelWorld world(store,{});GpuVoxelMotion motion(assets,count);
        FinishHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(upload.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        FinishHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
        const std::array<std::uint32_t,4> upstreamData{upstream,0,0,0};
        FinishHandle status(bgfx::createDynamicIndexBuffer(bgfx::copy(upstreamData.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        motion.finish(0,state.value,materialBuffer.value,3,status.value,world,store,{});
        if(repeat)motion.finish(0,motion.output(),materialBuffer.value,3,status.value,world,store,{});
        FinishHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        FinishHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        FinishHandle oldImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        FinishHandle oldStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        FinishHandle statsImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        FinishHandle statsStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count)for(auto pair:{std::pair{motion.output(),image.value},std::pair{state.value,oldImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,motion.status(),bgfx::Access::Read);bgfx::setImage(1,statsImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statExporter.value,1);
        std::vector<ParticleGpu> actual(storage),old(storage);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=statsImage.value;dst.handle=statsStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count){src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,old.data()));}
        while(bgfx::frame()<ready)SDL_PumpEvents();
        if(stats[0]!=expectedError)throw std::runtime_error("Voxel finish status case "+std::to_string(cases)+": "+std::to_string(stats[0])+" expected "+std::to_string(expectedError));
        unsigned caps=0,totalContacts=0;
        for(unsigned i=0;i<count;++i) {
            require(!std::memcmp(&input[i],&old[i],64),"Voxel finish mutated input");const auto& before=input[i];const auto& after=actual[i];
            if(expectedError || !(before.identity[1]&1) || before.identity[0]==2 || before.velocityInvMass[3]==0){require(!std::memcmp(&before,&after,64),"Voxel finish rollback/excluded slot changed");continue;}
            require(!std::memcmp(before.positionRadius,after.positionRadius,16) && before.velocityInvMass[3]==after.velocityInvMass[3] && !std::memcmp(before.thermalState,after.thermalState,32),"Voxel finish changed nonvelocity state");
            Vector p{before.positionRadius[0],before.positionRadius[1],before.positionRadius[2]},v{before.velocityInvMass[0],before.velocityInvMass[1],before.velocityInvMass[2]},result{after.velocityInvMass[0],after.velocityInvMass[1],after.velocityInvMass[2]};
            const auto shell=contacts(p,occupied,own,terrainFriction);totalContacts+=unsigned(shell.size());bool capped=false;auto expected=referenceFinish(v,shell,&capped);
            if(capped)++caps;
            for(unsigned a=0;a<3;++a)if(std::abs(result[a]-expected[a])>8e-4)throw std::runtime_error("Voxel finish CPU case "+std::to_string(cases)+" slot "+std::to_string(i)+" axis "+std::to_string(a)+": "+std::to_string(result[a])+" expected "+std::to_string(expected[a]));
            if(own==0 && std::abs(v[1]-1e-6)<1e-12)require(std::abs(result[1]-1e-6)<1e-10,"Voxel finish erased small free tangent");
            require(std::hypot(result[0],result[1],result[2])<=std::min(12.0,std::hypot(v[0],v[1],v[2]))+2e-5,"Voxel finish gained energy");
            for(auto c:shell)require(dot(result,c.normal)>=-1e-5,"Voxel finish left inward velocity");
        }
        if(!expectedError){require(stats[1]==(repeat?0:caps) && stats[3]==totalContacts,"Voxel finish counters disagree");if(repeat)require(stats[2]==0,"Voxel finish applied repeated friction");}
        const auto previous=motion.output();auto reject=[&](const SparseVoxelStore& source,Chunk origin){bool caught=false;try{motion.finish(0,state.value,materialBuffer.value,3,status.value,world,source,origin);}catch(const std::invalid_argument&){caught=true;}require(caught && motion.output().idx==previous.idx,"Voxel finish accepted stale world");};
        reject(store,{1,0,0});SparseVoxelStore foreign(ResidentChunkLayout({-4,0,-4},{8,4,8}),3);reject(foreign,{});
        store.setMaterial(address({15,15,15}),0,store.revision());reject(store,{});
        ++cases;particles+=count;
    };
    check({},{});
    for(double x:{-12.0,0.0,11.875})for(unsigned axis=0;axis<3;++axis)for(double sign:{-1.0,1.0}) {
        Vector lower{x,2,0},p{x+.0625,2.0625,.0625},v{};p[axis]=lower[axis]+(sign>0?.125:0)+sign*(.0625+2e-6);v[axis]=-sign*2;v[(axis+1)%3]=3;
        check({lower},{body(p,v)});check({lower},{body(p,v)},0,1);
        v[axis]=sign*2;check({lower},{body(p,v)});
    }
    // Simultaneous orthogonal surfaces, free Z slide, and idempotent finishing.
    const std::vector<Vector> corner{{0,2,0},{-.125,2.125,0}};
    check(corner,{body({-.062502,2.062498,.0625},{2,3,5})});
    check(corner,{body({-.062502,2.062498,.0625},{2,3,5})},0,0);
    check(corner,{body({-.062502,2.062498,.0625},{2,3,5})},.4f,.9f,0,0,true);
    check({}, {body({-15.9375,.0625,-15.9375},{-2,-3,-4})});
    check({}, {body({-15.9375,2,0},{-2,3,0})});
    check({{-16,2,0}},{body({-15.9375,2.187502,.0625},{-2,-3,5})});
    check({{0,2,0}},{body({-.062506,2.0625,.0625},{2,3,0})}); // outside shell
    check({{0,2,0}},{body({-.0625,2.0625,.0625},{0,3,0})}); // tangent
    check({{0,2,0}},{body({-.062502,2.0625,.0625},{2,1e-6,0})},0,0);
    check({{-5.875,3.75,-7.25},{-5.625,3.75,-7.25}},
          {body({-5.6875,3.87525,-7.1875},{0,-1,1})},0,0);
    check({{0,2,0}},{body({-.062502,2.0625,.0625},{1e30,1,0})},0,0);
    check({}, {body({-11,2,0},{1e30,2e30,-3e30})});
    check({{0,2,0}},{body({-.062502,2.0625,.0625},{1e30,2e30,-3e30})});
    auto invalid=body({-.06,2.0625,.0625},{1,2,3}),good=body({-11,2,0},{1,2,3});
    check({{0,2,0}},{invalid,good},.4f,.9f,2);
    check({}, {good},.4f,.9f,1,8);
    check({}, {good},-std::numeric_limits<float>::denorm_min(),.9f,2);
    check({{0,2,0}},{body({-.062502,2.0625,.0625},{1,2,3})},.4f,1.1f,2);
    check({{0,2,0}},{body({-.062502,2.0625,.0625},{1,2,3})},.4f,.9f,2,0,false,2);
    invalid=good;invalid.velocityInvMass[0]=std::numeric_limits<float>::infinity();check({}, {invalid,good},.4f,.9f,2);
    invalid=good;invalid.identity[3]=0;check({}, {invalid,good},.4f,.9f,2);
    auto excluded=body({.0625,2.0625,.0625},{1,2,3});excluded.identity[0]=2;auto dead=excluded;dead.identity[1]=0;auto fixed=excluded;fixed.identity[0]=0;fixed.velocityInvMass[3]=0;
    check({{0,2,0}},{excluded,dead,fixed});
    std::mt19937 random(18371);std::uniform_real_distribution<double> coordinate(.1,1),velocity(-8,8);
    std::vector<Vector> occupied;std::vector<ParticleGpu> batch;
    for(unsigned i=0;i<256;++i) {
        Vector lower{-12.0+double(i%16),2,-7.0+double(i/16)},n,p,v;
        for(unsigned a=0;a<3;++a)n[a]=coordinate(random)*((i&(1u<<a))?1:-1);const double magnitude=std::sqrt(dot(n,n));
        for(unsigned a=0;a<3;++a){n[a]/=magnitude;p[a]=lower[a]+(n[a]>0?.125:0)+n[a]*(.0625+2e-6);v[a]=velocity(random);}
        occupied.push_back(lower);batch.push_back(body(p,v));
    }
    check(occupied,batch);check(occupied,batch,0,0);
    std::cout<<"GPU voxel velocity finish passed: "<<cases<<" fixtures / "<<particles<<" particles; CPU cone/friction, simultaneous contacts, energy, immutable positions, rollback and revision gates\n";
}
}
