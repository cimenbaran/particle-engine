#include "GpuJacobiContacts.hpp"
#include "../../tests/reference/PositionFriction.hpp"
#include "../../tests/reference/SweptSphere.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/SpatialGrid.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace pw {
namespace {
template<class T> struct ContactHandle {
    T value;
    explicit ContactHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Contact test allocation failed");}
    ~ContactHandle(){bgfx::destroy(value);}
    ContactHandle(const ContactHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
struct ReferenceStep {std::vector<ParticleGpu> state;std::uint32_t error=0,maxContacts=0;};
ReferenceStep jacobiReference(const std::vector<ParticleGpu>& input,float relaxation,float limit,bool bounded=false,const std::vector<ParticleGpu>* start=nullptr,const std::array<MaterialGpu,4>* materials=nullptr,FloorTractionProfile floor={},AnalyticSphere obstacle={},GranularGridProfile profile=GranularGridProfile::Legacy) {
    ReferenceStep result{input};const auto count=std::uint32_t(input.size());
    const bool basin=profile==GranularGridProfile::Basin;
    const float wall=basin?15.9375f:7.9375f;
    const SpatialGrid grid({0,0,0},0.125f,{basin?-128:-64,0,basin?-128:-64},{basin?256u:128u,128,basin?256u:128u});
    std::vector<bool> included(count);
    for(std::uint32_t i=0;i<count;++i) {
        const auto& p=input[i];if(!(p.identity[1]&1)) continue;
        if(p.identity[0]>=4) {result.error|=1;continue;}
        if(p.identity[0]==1 || p.identity[0]==2) continue;
        included[i]=true;
        if(p.identity[2]!=i || !p.identity[3] || !grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]})) result.error|=1;
    }
    if(result.error) return result;
    for(std::uint32_t i=0;i<count;++i) if(included[i]) {
        const auto& p=input[i];const auto mass=p.velocityInvMass[3];
        if(p.positionRadius[3]!=GpuJacobiContacts::radius || !std::isfinite(mass) || mass<0 || (mass!=0 && mass<std::numeric_limits<float>::min())) result.error|=4;
    }
    if(result.error) return result;
    if(start) for(std::uint32_t i=0;i<count;++i) if(included[i]) {
        const auto& old=(*start)[i];const auto& p=input[i];const float mu=(*materials)[p.identity[0]].mechanical[1];
        if(std::memcmp(old.identity,p.identity,16) || old.positionRadius[3]!=p.positionRadius[3] ||
           std::memcmp(&old.velocityInvMass[3],&p.velocityInvMass[3],4) ||
           !grid.keyAt({old.positionRadius[0],old.positionRadius[1],old.positionRadius[2]}) ||
           !std::isfinite(mu) || mu<0 || mu>1) result.error|=32;
        if(floor.gravity>0){
            const float drag=(*materials)[p.identity[0]].mechanical[3];
            if(!std::isfinite(drag) || drag<0 || drag>100)result.error|=32;
            for(unsigned axis=0;axis<3;++axis)if(!std::isfinite(old.velocityInvMass[axis]))result.error|=32;
        }
    }
    if(result.error)return result;
    if(bounded) for(std::uint32_t i=0;i<count;++i) if(included[i] && input[i].velocityInvMass[3]!=0)
        for(unsigned axis=0;axis<3;++axis) if(input[i].positionRadius[axis]<(axis==1?0.0625f:-wall) || input[i].positionRadius[axis]>(axis==1?15.9375f:wall)) result.error|=16;
    if(obstacle.radius>0)for(unsigned i=0;i<count;++i)if(included[i] && input[i].velocityInvMass[3]!=0){
        double squared=0;for(unsigned axis=0;axis<3;++axis){double d=double(input[i].positionRadius[axis])-obstacle.centre[axis];squared+=d*d;}
        if(std::sqrt(squared)<obstacle.radius+0.0625)result.error|=16;
        if(start){double oldSquared=0;for(unsigned axis=0;axis<3;++axis){const double d=double((*start)[i].positionRadius[axis])-obstacle.centre[axis];oldSquared+=d*d;}
            if(std::sqrt(oldSquared)<obstacle.radius+0.0625)result.error|=32;}
    }
    if(result.error) return result;
    for(std::uint32_t i=0;i<count;++i) if(included[i]) {
        const auto& a=input[i];std::array<double,3> correction{};std::uint32_t contacts=0;
        for(std::uint32_t j=0;j<count;++j) if(j!=i && included[j]) {
            const auto& b=input[j];std::array<double,3> delta{};double squared=0;
            for(unsigned axis=0;axis<3;++axis) {delta[axis]=double(b.positionRadius[axis])-a.positionRadius[axis];squared+=delta[axis]*delta[axis];}
            const double distance=std::sqrt(squared),penetration=2*GpuJacobiContacts::radius-distance;
            if(penetration<=0) continue;
            ++contacts;const double wi=a.velocityInvMass[3],wj=b.velocityInvMass[3];if(wi==0 || wi+wj==0) continue;
            std::array<double,3> direction{};
            if(squared>1e-12) for(unsigned axis=0;axis<3;++axis) direction[axis]=-delta[axis]/distance;
            else direction[(std::min(i,j)*1664525u+std::max(i,j)*1013904223u)%3]=i<j?-1:1;
            reference::FrictionVector friction{};
            if(start && squared>1e-12) {
                reference::FrictionVector relative{};
                for(unsigned axis=0;axis<3;++axis)relative[axis]=(double(a.positionRadius[axis])-(*start)[i].positionRadius[axis])-(double(b.positionRadius[axis])-(*start)[j].positionRadius[axis]);
                const double mu=std::sqrt(double((*materials)[a.identity[0]].mechanical[1])*(*materials)[b.identity[0]].mechanical[1]);
                friction=reference::positionFriction(relative,direction,penetration,mu,0);
            }
            for(unsigned axis=0;axis<3;++axis) correction[axis]+=(direction[axis]*penetration+friction[axis])*wi/(wi+wj);
        }
        result.maxContacts=std::max(result.maxContacts,contacts);
        double squared=0;for(auto& x:correction) {x*=relaxation;squared+=x*x;}
        if(squared>double(limit)*limit) for(auto& x:correction) x*=limit/std::sqrt(squared);
        for(unsigned axis=0;axis<3;++axis) result.state[i].positionRadius[axis]=float(double(a.positionRadius[axis])+correction[axis]);
        if(bounded && a.velocityInvMass[3]!=0) {
            reference::FrictionVector projected{},normalDelta{},tangent{};
            double depth=0;
            for(unsigned axis=0;axis<3;++axis){
                const auto position=result.state[i].positionRadius[axis];projected[axis]=std::clamp(position,axis==1?0.0625f:-wall,axis==1?15.9375f:wall);
                normalDelta[axis]=projected[axis]-position;depth+=normalDelta[axis]*normalDelta[axis];
            }
            depth=std::sqrt(depth);
            if(start) {
                const auto& previous=(*start)[i];const auto& material=(*materials)[a.identity[0]];
                double remaining=0;
                if(floor.gravity>0 && a.positionRadius[1]==0.0625f && previous.positionRadius[1]==0.0625f && std::abs(previous.velocityInvMass[1])<=floor.speedLimit &&
                   normalDelta[0]==0 && normalDelta[2]==0 && std::abs(previous.velocityInvMass[0])<=floor.speedLimit && std::abs(previous.velocityInvMass[2])<=floor.speedLimit){
                    const double damping=std::max(0.0,1.0-double(material.mechanical[3])*floor.substepSeconds);
                    double tangentSpeed=std::hypot(double(previous.velocityInvMass[0]),double(previous.velocityInvMass[2]))*damping;
                    double normalSpeed=std::max(0.0,double(floor.gravity)*floor.substepSeconds-previous.velocityInvMass[1])*damping;
                    const double speed=std::hypot(tangentSpeed,normalSpeed);
                    if(speed>floor.speedLimit){const double scale=double(floor.speedLimit)/speed*0.9999995231628418;tangentSpeed*=scale;normalSpeed*=scale;}
                    remaining=reference::floorStaticBudget(tangentSpeed,normalSpeed,floor.substepSeconds,material.mechanical[1]);
                }
                double lengthSquared=0;
                for(unsigned axis=0;axis<3;++axis){tangent[axis]=(normalDelta[axis]!=0 || (axis==1 && remaining>0))?0:projected[axis]-previous.positionRadius[axis];lengthSquared+=tangent[axis]*tangent[axis];}
                if(std::sqrt(lengthSquared)<=double(material.mechanical[1])*depth+remaining)
                    for(unsigned axis=0;axis<3;++axis)projected[axis]-=tangent[axis];
            }
            double combined=0;
            for(unsigned axis=0;axis<3;++axis){projected[axis]=std::clamp(projected[axis],axis==1?0.0625:-double(wall),axis==1?15.9375:double(wall));correction[axis]=projected[axis]-a.positionRadius[axis];combined+=correction[axis]*correction[axis];}
            if(combined>double(limit)*limit)for(auto& value:correction)value*=limit/std::sqrt(combined);
            for(unsigned axis=0;axis<3;++axis)result.state[i].positionRadius[axis]=float(double(a.positionRadius[axis])+correction[axis]);
        }
        if(obstacle.radius>0 && a.velocityInvMass[3]!=0){
            reference::SweepState path;for(unsigned axis=0;axis<3;++axis){path.position[axis]=a.positionRadius[axis];path.velocity[axis]=double(result.state[i].positionRadius[axis])-a.positionRadius[axis];}
            const reference::SphereCollider sphere{{obstacle.centre[0],obstacle.centre[1],obstacle.centre[2]},obstacle.radius};
            const auto swept=reference::sweepSphere(path,1,a.positionRadius[3],sphere,0,0,0);
            if(swept.hit){
                double combined=0;for(unsigned axis=0;axis<3;++axis){correction[axis]=swept.state.position[axis]+swept.normal[axis]*0.000002-a.positionRadius[axis];combined+=correction[axis]*correction[axis];}
                if(combined>double(limit)*limit)for(auto& value:correction)value*=limit/std::sqrt(combined);
                for(unsigned axis=0;axis<3;++axis)result.state[i].positionRadius[axis]=float(double(a.positionRadius[axis])+correction[axis]);
            }
        }
        const auto& p=result.state[i];if(!grid.keyAt({p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]})) result.error|=16;
    }
    if(result.error) result.state=input;
    return result;
}
}
void runGpuJacobiContactsTest(const std::filesystem::path& assets) {
    std::array<MaterialGpu,4> materials{};for(unsigned i=0;i<4;++i) {materials[i].identity[0]=i;materials[i].identity[1]=i==3?0:i;}
    materials[0].mechanical[1]=0.5f;materials[3].mechanical[1]=0.25f;
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    ContactHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
    ContactHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    ContactHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> input,unsigned iterations=1,float relaxation=0.8f,float limit=GpuJacobiContacts::radius*0.5f,GpuJacobiContacts* reused=nullptr,bool bounded=false,const std::vector<ParticleGpu>* baseline=nullptr,FloorTractionProfile floor={},AnalyticSphere obstacle={},GranularGridProfile profile=GranularGridProfile::Legacy) {
        const float wall=profile==GranularGridProfile::Basin?15.9375f:7.9375f;
        const auto count=std::uint32_t(input.size());
        auto owned=reused?nullptr:std::make_unique<GpuJacobiContacts>(assets,count,relaxation,limit,bounded,obstacle,profile);
        auto& solver=reused?*reused:*owned;
        ContactHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        ContactHandle oldState(bgfx::createDynamicVertexBuffer(bgfx::copy(baseline?baseline->data():input.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        ContactHandle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ContactHandle originalImage(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle originalStaging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ContactHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        auto expected=input;ReferenceStep last;auto current=state.value;
        for(unsigned iteration=0;iteration<iterations;++iteration) {
            last=jacobiReference(expected,relaxation,limit,bounded,baseline,&materials,floor,obstacle,profile);expected=last.state;
            solver.submit(0,current,materialBuffer.value,std::uint32_t(materials.size()),baseline?oldState.value:bgfx::DynamicVertexBufferHandle{bgfx::kInvalidHandle},floor);current=solver.output();
        }
        const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);bgfx::setBuffer(0,current,bgfx::Access::Read);
        bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setImage(1,originalImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        bgfx::setBuffer(0,solver.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(count),original(count);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,actual.data());
        src.handle=originalImage.value;dst.handle=originalStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
        src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,stats.data()));
        while(bgfx::frame()<ready) SDL_PumpEvents();
        require(std::memcmp(original.data(),input.data(),count*sizeof(ParticleGpu))==0,"Jacobi mutated its immutable input");
        if(stats[0]!=last.error) throw std::runtime_error("Jacobi error mismatch actual "+std::to_string(stats[0])+" expected "+std::to_string(last.error));
        if(!last.error && iterations==1) require(stats[1]==last.maxContacts,"Jacobi overlap metric mismatch");
        require(stats[1]<count && stats[2]<=count && stats[3]<=27*64,"Jacobi diagnostics outside bounds");
        for(std::uint32_t i=0;i<count;++i) {
            require(std::memcmp(&actual[i].positionRadius[3],&input[i].positionRadius[3],sizeof(ParticleGpu)-3*sizeof(float))==0,"Jacobi changed radius, velocity, thermal state or identity");
            if(last.error || !(input[i].identity[1]&1) || input[i].identity[0]==1 || input[i].identity[0]==2 || input[i].velocityInvMass[3]==0) {
                require(std::memcmp(&actual[i],&input[i],sizeof(ParticleGpu))==0,"Jacobi modified failed/excluded/static state");continue;
            }
            double displacement=0;
            for(unsigned axis=0;axis<3;++axis) {
                require(std::isfinite(actual[i].positionRadius[axis]) && std::abs(actual[i].positionRadius[axis]-expected[i].positionRadius[axis])<2e-5f,"Jacobi CPU-reference position mismatch");
                if(bounded) require(actual[i].positionRadius[axis]>=(axis==1?0.0625f:-wall) &&
                    actual[i].positionRadius[axis]<=(axis==1?15.9375f:wall),"Bounded Jacobi escaped inset planes");
                const double d=double(actual[i].positionRadius[axis])-input[i].positionRadius[axis];displacement+=d*d;
            }
            if(obstacle.radius>0){double squared=0;for(unsigned axis=0;axis<3;++axis){const double d=double(actual[i].positionRadius[axis])-obstacle.centre[axis];squared+=d*d;}
                require(std::sqrt(squared)>=obstacle.radius+0.0625-1e-7,"Jacobi correction penetrated sphere");}
            require(std::sqrt(displacement)<=double(iterations)*limit+2e-6,"Jacobi correction bound exceeded");
        }
        if(baseline) {
            bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,oldState.value,bgfx::Access::Read);
            bgfx::setImage(1,originalImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            src.handle=originalImage.value;dst.handle=originalStaging.value;bgfx::blit(2,dst,src);ready=bgfx::read(dst,original.data());
            while(bgfx::frame()<ready)SDL_PumpEvents();
            require(std::memcmp(original.data(),baseline->data(),count*sizeof(ParticleGpu))==0,"Jacobi mutated substep baseline");
        }
        ++cases;
    };
    auto fixture=[](std::uint32_t count,bool dense=false) {
        std::vector<ParticleGpu> p(count);
        for(std::uint32_t i=0;i<count;++i) {
            p[i].positionRadius[0]=dense?0.f:float(i%8)*0.046875f;
            p[i].positionRadius[1]=4.f+(dense?0.f:float((i/8)%8)*0.046875f);
            p[i].positionRadius[2]=dense?0.f:float(i/64)*0.046875f;
            p[i].positionRadius[3]=GpuJacobiContacts::radius;p[i].velocityInvMass[0]=0.25f;p[i].velocityInvMass[3]=1;
            p[i].thermalState[0]=300;p[i].thermalState[1]=0.5f;p[i].identity[0]=i%2?3:0;
            p[i].identity[1]=1;p[i].identity[2]=i;p[i].identity[3]=i+1;
        }return p;
    };
    for(std::uint32_t count:{1u,2u,127u,128u,129u,255u,256u,257u}) check(fixture(count));
    for(std::uint32_t count:{2u,129u,257u}) {check(fixture(count,true));check(fixture(count),8);}
    auto pair=fixture(2);pair[1].positionRadius[0]=0.0625f;
    check(pair,1,1,0.0625f);pair[0].velocityInvMass[3]=0.25f;check(pair,1,1,0.0625f);
    pair[0].velocityInvMass[3]=0;check(pair,1,1,0.0625f);pair[1].velocityInvMass[3]=0;check(pair);
    pair=fixture(2);pair[1].positionRadius[0]=0.125f;check(pair);pair[1].positionRadius[0]=1;check(pair);
    auto mixed=fixture(129);for(unsigned i=0;i<mixed.size();++i) {
        if(i%7==1) mixed[i].identity[0]=1;
        if(i%7==2) mixed[i].identity[0]=2;
        if(i%7==3) {mixed[i].identity[1]=0;mixed[i].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();}
        if(i%7==4) mixed[i].velocityInvMass[3]=0;
    }check(mixed);check(mixed,4);
    auto bad=fixture(3);bad[0].positionRadius[3]=0.05f;check(bad);
    for(float mass:{-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::denorm_min()}) {bad=fixture(3);bad[0].velocityInvMass[3]=mass;check(bad);}
    bad=fixture(3);bad[0].positionRadius[0]=8;check(bad);
    bad=fixture(3);bad[0].identity[3]=0;check(bad);
    bad=fixture(2);bad[0].positionRadius[1]=0;bad[1].positionRadius[1]=0.03125f;check(bad);
    pair=fixture(2);pair[0].velocityInvMass[3]=std::numeric_limits<float>::max();pair[1].velocityInvMass[3]=std::numeric_limits<float>::max();check(pair);
    pair[0].velocityInvMass[3]=std::numeric_limits<float>::min();pair[1].velocityInvMass[3]=std::numeric_limits<float>::min();check(pair);
    pair[1].velocityInvMass[3]=std::numeric_limits<float>::max();check(pair);
    std::swap(pair[0].velocityInvMass[3],pair[1].velocityInvMass[3]);check(pair);
    // Opposed pair corrections into every inset plane, including repeated
    // rebuilds and a corner. Projection must retain the displacement bound.
    for(unsigned axis=0;axis<3;++axis) for(bool high:{false,true}) {
        pair=fixture(2);pair[0].positionRadius[0]=pair[1].positionRadius[0]=0;
        const float wall=axis==1?(high?15.9375f:0.0625f):(high?7.9375f:-7.9375f);
        pair[0].positionRadius[axis]=wall;pair[1].positionRadius[axis]=wall+(high?-0.0625f:0.0625f);
        check(pair,1,0.8f,0.03125f,nullptr,true);check(pair,8,0.8f,0.03125f,nullptr,true);
    }
    pair=fixture(2);pair[0].positionRadius[0]=pair[0].positionRadius[2]=-7.9375f;
    pair[1].positionRadius[0]=pair[1].positionRadius[2]=-7.90625f;check(pair,8,0.8f,0.03125f,nullptr,true);
    bad=fixture(3);bad[0].positionRadius[1]=0.03125f;check(bad,1,0.8f,0.03125f,nullptr,true);
    bad[0].velocityInvMass[3]=0;check(bad,1,0.8f,0.03125f,nullptr,true);
    GpuJacobiContacts recovering(assets,3);
    bad=fixture(3);bad[0].positionRadius[3]=0.05f;
    check(bad,1,0.8f,0.03125f,&recovering);check(fixture(3),1,0.8f,0.03125f,&recovering);
    {
        GpuJacobiContacts zero(assets,0);const ParticleGpu dummy{};
        ContactHandle state(bgfx::createDynamicVertexBuffer(bgfx::copy(&dummy,sizeof(dummy)),layout,BGFX_BUFFER_COMPUTE_READ));
        ContactHandle image(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ContactHandle staging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        zero.submit(0,state.value,materialBuffer.value,std::uint32_t(materials.size()));
        bgfx::setBuffer(0,zero.status(),bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::array<std::uint32_t,4> status{};bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;
        bgfx::blit(2,dst,src);const auto ready=bgfx::read(dst,status.data());while(bgfx::frame()<ready) SDL_PumpEvents();
        require(status==std::array<std::uint32_t,4>{0,0,0,0},"Zero-capacity Jacobi iteration failed");++cases;
    }
    auto empty=fixture(129);for(auto& p:empty) p.identity[1]=0;check(empty);
    pair=fixture(2);pair[1].positionRadius[0]=0.0625f;auto baseline=pair;
    baseline[0].positionRadius[1]-=0.001953125f;
    check(pair,1,1,0.0625f,nullptr,false,&baseline);check(pair,4,0.8f,0.03125f,nullptr,true,&baseline);
    baseline=pair;baseline[0].positionRadius[1]-=0.125f;check(pair,1,1,0.0625f,nullptr,false,&baseline); // sliding: no extra position friction
    baseline=pair;for(auto& body:baseline)body.positionRadius[1]-=0.001953125f;check(pair,1,1,0.0625f,nullptr,false,&baseline);
    for(unsigned axis:{1u,2u})for(float inverseMass:{0.0f,0.25f,1.0f,4.0f}) {
        pair=fixture(2);pair[1].positionRadius[0]=0.0625f;pair[1].velocityInvMass[3]=inverseMass;baseline=pair;
        baseline[0].positionRadius[axis]-=0.001953125f;check(pair,1,1,0.0625f,nullptr,false,&baseline);
    }
    pair=fixture(129);baseline=pair;for(unsigned i=0;i<129;++i)baseline[i].positionRadius[1]-=float(i%3)*0.001953125f;
    check(pair,4,0.8f,0.03125f,nullptr,true,&baseline);
    for(unsigned axis=0;axis<3;++axis)for(bool high:{false,true}) {
        auto wall=fixture(2);wall[0].identity[0]=wall[1].identity[0]=0;
        wall[0].positionRadius[0]=wall[1].positionRadius[0]=0;
        const float face=axis==1?(high?15.9375f:0.0625f):(high?7.9375f:-7.9375f);
        wall[0].positionRadius[axis]=face;wall[1].positionRadius[axis]=face+(high?-0.0625f:0.0625f);
        auto old=wall;const unsigned tangent=(axis+1)%3;old[0].positionRadius[tangent]-=0.001953125f;
        check(wall,1,1,0.0625f,nullptr,true,&old);
    }
    for(unsigned invalid=0;invalid<5;++invalid){baseline=pair;
        if(invalid==0)baseline[7].identity[3]++;
        if(invalid==1)baseline[7].velocityInvMass[3]=2;
        if(invalid==2)baseline[7].positionRadius[3]=0.05f;
        if(invalid==3)baseline[7].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();
        if(invalid==4)baseline[7].positionRadius[0]=9;
        check(pair,1,0.8f,0.03125f,nullptr,true,&baseline);
    }
    // Integer identity comparisons must not round adjacent generations together.
    for(std::uint32_t generation:{16777216u,2147483648u,UINT32_MAX-1}) {
        auto current=fixture(2);current[0].identity[3]=generation;auto previous=current;
        check(current,1,.8f,.03125f,nullptr,true,&previous);
        ++previous[0].identity[3];check(current,1,.8f,.03125f,nullptr,true,&previous);
    }
    pair=fixture(2);pair[1].positionRadius[0]=0.0625f;baseline=pair;baseline[0].positionRadius[1]-=0.001953125f;
    for(float mu:{0.0f,1.0f,-1.0f,1.1f,-std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        materials[0].mechanical[1]=mu;bgfx::update(materialBuffer.value,0,bgfx::copy(materials.data(),sizeof(materials)));
        check(pair,1,1,0.0625f,nullptr,false,&baseline);
    }
    materials[0].mechanical[1]=0.5f;bgfx::update(materialBuffer.value,0,bgfx::copy(materials.data(),sizeof(materials)));
    auto corner=fixture(3);for(auto& body:corner){body.identity[0]=0;body.positionRadius[0]=body.positionRadius[2]=-7.9375f;body.positionRadius[1]=4;}
    corner[1].positionRadius[0]+=0.0625f;corner[2].positionRadius[2]+=0.0625f;
    baseline=corner;baseline[0].positionRadius[1]-=0.001953125f;check(corner,1,1,0.03125f,nullptr,true,&baseline);
    // Gravity support already consumed by swept sliding must not be granted twice.
    for(float speed:{0.0f,0.01f,0.03f,1.0f})for(unsigned iterations:{1u,4u}){
        pair=fixture(1);pair[0].positionRadius[0]=0.00025f;pair[0].positionRadius[1]=0.0625f;baseline=pair;
        baseline[0].positionRadius[0]=0;baseline[0].velocityInvMass[0]=speed;baseline[0].velocityInvMass[1]=baseline[0].velocityInvMass[2]=0;
        check(pair,iterations,1,0.03125f,nullptr,true,&baseline,{9.81f,0.01f,12});
    }
    for(float vertical:{-0.1f,0.00055f,0.04f,0.0981f,0.2f})for(unsigned iterations:{1u,4u}){
        pair=fixture(1);pair[0].positionRadius[0]=0.00025f;pair[0].positionRadius[1]=0.0625f;baseline=pair;
        baseline[0].positionRadius[0]=0;baseline[0].velocityInvMass[0]=baseline[0].velocityInvMass[2]=0;
        baseline[0].velocityInvMass[1]=vertical;
        check(pair,iterations,1,0.03125f,nullptr,true,&baseline,{9.81f,0.01f,12});
    }
    for(unsigned mode=0;mode<7;++mode){
        pair=fixture(1);pair[0].positionRadius[0]=0.00025f;pair[0].positionRadius[1]=0.0625f;baseline=pair;
        baseline[0].positionRadius[0]=0;for(unsigned axis=0;axis<3;++axis)baseline[0].velocityInvMass[axis]=0;
        FloorTractionProfile floor{9.81f,0.01f,12};
        if(mode==0)floor.gravity=0;
        if(mode==1)materials[0].mechanical[1]=0;
        if(mode==2)materials[0].mechanical[3]=100;
        if(mode==3)baseline[0].velocityInvMass[1]=-0.1f;
        if(mode==4)floor.speedLimit=0.01f;
        if(mode==5)baseline[0].velocityInvMass[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==6)materials[0].mechanical[3]=-1;
        bgfx::update(materialBuffer.value,0,bgfx::copy(materials.data(),sizeof(materials)));
        check(pair,1,1,0.03125f,nullptr,true,&baseline,floor);
        materials[0].mechanical[1]=0.5f;materials[0].mechanical[3]=0;
    }
    bgfx::update(materialBuffer.value,0,bgfx::copy(materials.data(),sizeof(materials)));
    const AnalyticSphere obstacle{{0,4,0},0.0625f};
    for(unsigned axis=0;axis<3;++axis)for(float sign:{-1.0f,1.0f})for(unsigned iterations:{1u,8u}){
        pair=fixture(2);for(auto& body:pair)for(unsigned a=0;a<3;++a)body.positionRadius[a]=obstacle.centre[a];
        pair[0].positionRadius[axis]+=sign*0.126f;pair[1].positionRadius[axis]+=sign*0.196f;
        check(pair,iterations,1,0.03125f,nullptr,true,nullptr,{},obstacle);
    }
    for(float angle:{0.0f,0.3f,1.2f,2.8f})for(unsigned iterations:{1u,4u}){
        pair=fixture(2);const float nx=std::cos(angle),nz=std::sin(angle);
        pair[0].positionRadius[0]=nx*0.126f;pair[0].positionRadius[1]=4;pair[0].positionRadius[2]=nz*0.126f;
        pair[1].positionRadius[0]=pair[0].positionRadius[0]+nx*0.07f-nz*0.03f;pair[1].positionRadius[1]=4;
        pair[1].positionRadius[2]=pair[0].positionRadius[2]+nz*0.07f+nx*0.03f;
        check(pair,iterations,1,0.03125f,nullptr,true,nullptr,{},obstacle);
    }
    for(float z:{0.1248f,0.126f}){
        pair=fixture(2);pair[0].positionRadius[0]=-0.015f;pair[1].positionRadius[0]=-0.08f;
        for(auto& body:pair){body.positionRadius[1]=4;body.positionRadius[2]=z;}
        // At z=.1248 both unconstrained endpoints are outside, but the chord
        // crosses the sphere. At z=.126 the complete segment misses it.
        check(pair,1,1,0.03125f,nullptr,true,nullptr,{},obstacle);
    }
    pair=fixture(2);pair[0].positionRadius[0]=-0.15625f;pair[1].positionRadius[0]=-0.21875f;
    for(auto& body:pair){body.positionRadius[1]=4;body.positionRadius[2]=0;}
    check(pair,1,1,0.03125f,nullptr,true,nullptr,{},obstacle); // exact endpoint
    for(unsigned count:{127u,128u,129u,255u,256u,257u}){
        auto dense=fixture(count);for(unsigned i=0;i<count;++i){dense[i].positionRadius[0]=-0.14f;dense[i].positionRadius[1]=4+float(i%5)*0.01f;dense[i].positionRadius[2]=float(i%7)*0.01f;}
        check(dense,1,0.8f,0.03125f,nullptr,true,nullptr,{},obstacle);
    }
    pair=fixture(2);for(auto& body:pair){body.positionRadius[0]=body.positionRadius[2]=0;body.positionRadius[1]=4;}
    check(pair,1,1,0.03125f,nullptr,true,nullptr,{},obstacle); // whole-pool overlap rollback
    for(auto& body:pair)body.velocityInvMass[3]=0;
    check(pair,1,1,0.03125f,nullptr,true,nullptr,{},obstacle); // immovable preservation
    pair=fixture(1);pair[0].positionRadius[0]=-0.126f;pair[0].positionRadius[1]=4;pair[0].positionRadius[2]=0;
    baseline=pair;baseline[0].positionRadius[0]=0;
    check(pair,1,1,0.03125f,nullptr,true,&baseline,{},obstacle); // invalid original snapshot
    for(float x:{-15.9375f,-11.5f,11.5f,15.875f}) {
        pair=fixture(2);pair[0].positionRadius[0]=x;pair[1].positionRadius[0]=x+0.0625f;
        baseline=pair;baseline[0].positionRadius[1]-=0.001953125f;
        check(pair,4,0.8f,0.03125f,nullptr,true,&baseline,{}, {},GranularGridProfile::Basin);
    }
    std::cout<<"GPU Jacobi immutable-input corrections passed: "<<cases<<" fixtures, mass weighting, coincident pairs, bounds, failure rollback and multi-iteration rebuilds\n";
}
}
