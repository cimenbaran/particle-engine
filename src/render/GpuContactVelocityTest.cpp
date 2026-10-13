#include "GpuContactVelocity.hpp"
#include "GpuJacobiContacts.hpp"
#include "GpuVelocityReconstruction.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include "../../tests/reference/ContactVelocity.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
namespace pw {
namespace {
template<class T> struct ResponseHandle {
    T value;
    explicit ResponseHandle(T h):value(h){if(!bgfx::isValid(h)) throw std::runtime_error("Response test allocation failed");}
    ~ResponseHandle(){bgfx::destroy(value);}
    ResponseHandle(const ResponseHandle&)=delete;
};
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
using Materials=std::array<MaterialGpu,5>;
}
void runGpuContactVelocityTest(const std::filesystem::path& assets) {
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    ResponseHandle exporter(loadCompute(assets,"cs_export")),statsExporter(loadCompute(assets,"cs_pool_stats"));
    ResponseHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));unsigned cases=0;
    auto check=[&](std::vector<ParticleGpu> predicted,std::vector<ParticleGpu> current,Materials materials,float bound=0,
                  std::uint32_t errors=0,std::uint32_t upstream=0,bool closed=false,GpuContactVelocity* reused=nullptr,bool chain=false,GranularGridProfile profile=GranularGridProfile::Legacy) {
        const auto count=std::uint32_t(current.size()),storage=std::max(1u,count);require(predicted.size()==current.size(),"Response fixture size mismatch");
        auto owned=reused?nullptr:std::make_unique<GpuContactVelocity>(assets,count,bound,0.0001f,0.5f,profile);auto& response=reused?*reused:*owned;
        auto predUpload=predicted,currentUpload=current;if(!count) {predUpload.resize(1);currentUpload.resize(1);}
        ResponseHandle pred(bgfx::createDynamicVertexBuffer(bgfx::copy(predUpload.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        ResponseHandle input(bgfx::createDynamicVertexBuffer(bgfx::copy(currentUpload.data(),storage*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
        ResponseHandle materialBuffer(bgfx::createDynamicVertexBuffer(bgfx::copy(materials.data(),sizeof(materials)),layout,BGFX_BUFFER_COMPUTE_READ));
        const std::array<std::uint32_t,4> up{upstream,0,0,0};
        ResponseHandle sourceStatus(bgfx::createDynamicIndexBuffer(bgfx::copy(up.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ));
        std::unique_ptr<GpuJacobiContacts> contacts;std::unique_ptr<GpuVelocityReconstruction> reconstruction;
        std::unique_ptr<ResponseHandle<bgfx::DynamicVertexBufferHandle>> previous;
        auto state=input.value;auto source=sourceStatus.value;
        if(chain) {
            require(count==2,"Response chain fixture requires a pair");auto old=predicted;
            for(auto& p:old) for(unsigned axis=0;axis<3;++axis) p.positionRadius[axis]-=p.velocityInvMass[axis]*0.125f;
            previous=std::make_unique<ResponseHandle<bgfx::DynamicVertexBufferHandle>>(bgfx::createDynamicVertexBuffer(bgfx::copy(old.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
            contacts=std::make_unique<GpuJacobiContacts>(assets,count,0.8f,0.03125f,false,AnalyticSphere{},profile);reconstruction=std::make_unique<GpuVelocityReconstruction>(assets,count);
            contacts->submit(0,pred.value,materialBuffer.value,std::uint32_t(materials.size()));
            reconstruction->submit(0,previous->value,contacts->output(),materialBuffer.value,std::uint32_t(materials.size()),contacts->status(),0.125f);
            state=reconstruction->output();source=reconstruction->status();current=predicted;
            current[0].positionRadius[0]-=0.025f;current[1].positionRadius[0]+=0.025f;
            for(unsigned i=0;i<2;++i) for(unsigned axis=0;axis<3;++axis) current[i].velocityInvMass[axis]=(current[i].positionRadius[axis]-old[i].positionRadius[axis])/0.125f;
        }
        reference::ContactVelocityResult expected;std::vector<reference::ContactVelocityPoint> points;
        if(!errors) {
            for(std::uint32_t i=0;i<count;++i) {
                reference::ContactVelocityPoint p;p.slot=i;
                const auto material=current[i].identity[0];
                p.active=(current[i].identity[1]&1) && material<materials.size() && materials[material].identity[1]==0;
                for(unsigned axis=0;axis<3;++axis) {
                    p.predictedPosition[axis]=predicted[i].positionRadius[axis];p.position[axis]=current[i].positionRadius[axis];
                    p.incomingVelocity[axis]=predicted[i].velocityInvMass[axis];p.velocity[axis]=current[i].velocityInvMass[axis];
                }
                p.inverseMass=current[i].velocityInvMass[3];
                if(material<materials.size()) {p.restitution=materials[material].mechanical[2];p.friction=materials[material].mechanical[1];}
                points.push_back(p);
            }
            expected=reference::contactVelocityResponse(points,0.5,double(0.0001f),bound);
        }
        response.submit(0,pred.value,state,materialBuffer.value,std::uint32_t(materials.size()),source);
        ResponseHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ResponseHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ResponseHandle predImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ResponseHandle predStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ResponseHandle currentImage(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ResponseHandle currentStaging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        ResponseHandle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ResponseHandle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
        if(count) for(auto pair:{std::pair{response.output(),image.value},std::pair{pred.value,predImage.value},std::pair{state,currentImage.value}}) {
            const float p[]{0,float(count),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);
            bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
        }
        bgfx::setBuffer(0,response.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statsExporter.value,1);
        std::vector<ParticleGpu> actual(storage),predActual(storage),currentActual(storage);std::array<std::uint32_t,4> stats{};
        bgfx::TextureRegion src,dst;src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,stats.data());
        if(count) {
            src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,actual.data()));
            src.handle=predImage.value;dst.handle=predStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,predActual.data()));
            src.handle=currentImage.value;dst.handle=currentStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,currentActual.data()));
        }
        while(bgfx::frame()<ready) SDL_PumpEvents();
        if(stats[0]!=errors) throw std::runtime_error("Contact velocity error mismatch fixture "+std::to_string(cases)+" actual "+std::to_string(stats[0])+" expected "+std::to_string(errors));
        if(!errors) require(stats[1]==(count?*std::max_element(expected.degrees.begin(),expected.degrees.end()):0),"GPU contact degree mismatch");
        require(stats[1]<std::max(1u,count) && stats[2]<=count && stats[3]<=2097152u*64,"GPU response diagnostics outside bounds");
        double beforeEnergy=0,afterEnergy=0;std::array<double,3> beforeMomentum{},afterMomentum{};
        for(std::uint32_t i=0;i<count;++i) {
            require(std::memcmp(&predActual[i],&predicted[i],sizeof(ParticleGpu))==0,"Response mutated incoming snapshot");
            if(!chain) require(std::memcmp(&currentActual[i],&current[i],sizeof(ParticleGpu))==0,"Response mutated reconstructed snapshot");
            const auto& baseline=currentActual[i];
            require(std::memcmp(actual[i].positionRadius,baseline.positionRadius,16)==0,"Response moved a particle");
            require(std::memcmp(&actual[i].velocityInvMass[3],&baseline.velocityInvMass[3],sizeof(ParticleGpu)-7*sizeof(float))==0,"Response changed mass, thermal state or identity");
            if(errors || !points[i].active || points[i].inverseMass==0) {require(std::memcmp(&actual[i],&baseline,sizeof(ParticleGpu))==0,"Response changed failed/static/excluded state");continue;}
            for(unsigned axis=0;axis<3;++axis) {
                const double wanted=expected.velocities[i][axis],value=actual[i].velocityInvMass[axis];
                if(!std::isfinite(value) || std::abs(value-wanted)>2e-5*std::max(1.,std::abs(wanted)))
                    throw std::runtime_error("GPU contact velocity CPU mismatch fixture "+std::to_string(cases)+" slot "+std::to_string(i)+" axis "+std::to_string(axis)+" actual "+std::to_string(value)+" expected "+std::to_string(wanted));
                if(closed) {
                    const double old=current[i].velocityInvMass[axis],mass=1/points[i].inverseMass;
                    beforeEnergy+=0.5*mass*old*old;afterEnergy+=0.5*mass*value*value;
                    beforeMomentum[axis]+=mass*old;afterMomentum[axis]+=mass*value;
                }
            }
        }
        if(closed) {
            require(afterEnergy<=beforeEnergy+1e-4*std::max(1.,beforeEnergy),"GPU closed collision response gained energy");
            for(unsigned axis=0;axis<3;++axis) require(std::abs(afterMomentum[axis]-beforeMomentum[axis])<=1e-4*std::max(1u,count),"GPU closed response lost momentum");
            if(count==2 && current[0].positionRadius[0]!=current[1].positionRadius[0] &&
               current[0].positionRadius[1]==current[1].positionRadius[1] && current[0].positionRadius[2]==current[1].positionRadius[2]) {
                double alignment=0,oldTangent=0,newTangent=0;
                for(unsigned axis=1;axis<3;++axis) {
                    const double a=current[0].velocityInvMass[axis]-current[1].velocityInvMass[axis];
                    const double b=actual[0].velocityInvMass[axis]-actual[1].velocityInvMass[axis];
                    alignment+=a*b;oldTangent+=a*a;newTangent+=b*b;
                }
                require(alignment>=-1e-6 && newTangent<=oldTangent+1e-6,"GPU isolated-pair friction reversed or increased sliding");
            }
        }
        if(chain) {require(std::abs(actual[0].velocityInvMass[0]+0.5f)<1e-5f && std::abs(actual[1].velocityInvMass[0]-0.5f)<1e-5f,"End-to-end pair restitution incorrect");}
        ++cases;
    };
    auto makeMaterials=[](float e,float mu) {Materials m{};for(unsigned i=0;i<5;++i) {m[i].identity[0]=i;m[i].identity[1]=i<2?0:i==2?1:i==3?2:99;m[i].mechanical[1]=mu;m[i].mechanical[2]=e;}return m;};
    auto fixture=[](std::uint32_t count,bool dense=false) {
        std::vector<ParticleGpu> p(count);
        for(std::uint32_t i=0;i<count;++i) {
            p[i].positionRadius[0]=dense?0.f:float(i%8)*0.046875f;p[i].positionRadius[1]=4+(dense?0.f:float((i/8)%8)*0.046875f);
            p[i].positionRadius[2]=dense?0.f:float(i/64)*0.046875f;p[i].positionRadius[3]=0.0625f;
            for(unsigned axis=0;axis<3;++axis) p[i].velocityInvMass[axis]=(int((i*(17+axis*2)+axis*7)%33)-16)*0.125f;
            p[i].velocityInvMass[3]=1;p[i].thermalState[0]=300;p[i].thermalState[3]=0.125f;p[i].identity[0]=i%2;
            p[i].identity[1]=1;p[i].identity[2]=i;p[i].identity[3]=3;
        }return p;
    };
    auto material=makeMaterials(0.25f,0.5f);
    for(std::uint32_t count:{0u,1u,127u,128u,129u,255u,256u,257u}) {
        auto p=fixture(count);for(std::uint32_t i=0;i<count;++i) p[i].velocityInvMass[3]=i%3==0?0.25f:i%3==1?1.f:4.f;
        check(p,p,material,0,0,0,true);
    }
    auto p=fixture(1024,true);check(p,p,material,0,0,0,true);
    auto pair=fixture(2);pair[1].positionRadius[0]=0.125f;
    for(float e:{0.f,0.25f,0.5f,1.f}) for(float mu:{0.f,0.5f,1.f}) for(float tangent:{0.f,0.01f,1.f}) {
        pair[0].velocityInvMass[0]=1;pair[1].velocityInvMass[0]=-1;pair[0].velocityInvMass[1]=tangent;pair[1].velocityInvMass[1]=-tangent;
        pair[0].velocityInvMass[2]=pair[1].velocityInvMass[2]=0;check(pair,pair,makeMaterials(e,mu),0,0,0,true);
    }
    pair[0].velocityInvMass[0]=0.125f;pair[1].velocityInvMass[0]=-0.125f;pair[0].velocityInvMass[1]=pair[1].velocityInvMass[1]=0;check(pair,pair,makeMaterials(1,0),0,0,0,true);
    pair[0].velocityInvMass[0]=-1;pair[1].velocityInvMass[0]=1;check(pair,pair,material,0,0,0,true);
    pair[0].velocityInvMass[0]=1;pair[1].velocityInvMass[0]=-1;
    auto mixed=makeMaterials(1,1);mixed[0].mechanical[1]=0.25f;mixed[1].mechanical[2]=0.25f;check(pair,pair,mixed,0,0,0,true);
    pair[0].velocityInvMass[3]=0.25f;check(pair,pair,material,0,0,0,true);pair[0].velocityInvMass[3]=1;
    pair[1].velocityInvMass[3]=0;pair[1].velocityInvMass[0]=0;check(pair,pair,material);pair[0].velocityInvMass[3]=0;check(pair,pair,material);
    pair=fixture(2);pair[1].positionRadius[0]=0.0625f;pair[0].velocityInvMass[0]=1;pair[1].velocityInvMass[0]=-1;
    for(auto& q:pair) q.velocityInvMass[1]=q.velocityInvMass[2]=0;
    check(pair,pair,makeMaterials(0.5f,0),0.25f,0,0,false,nullptr,true);
    auto predicted=pair,current=pair;for(auto& q:predicted) q.velocityInvMass[0]=0;
    current[0].velocityInvMass[0]=-2;current[1].velocityInvMass[0]=2;current[1].positionRadius[0]=0.125f;check(predicted,current,material,0.125f);
    predicted=pair;predicted[1].positionRadius[0]=0.125f;current=predicted;current[0].positionRadius[0]=-0.25f;current[1].positionRadius[0]=0.375f;check(predicted,current,material,0.25f);
    current=predicted;predicted[1].positionRadius[0]=0.25f;check(predicted,current,material,0.125f);
    auto excluded=fixture(129);for(unsigned i=0;i<excluded.size();++i) {
        if(i%5==1) excluded[i].identity[0]=2;if(i%5==2) excluded[i].identity[0]=3;
        if(i%5==3) {excluded[i].identity[1]=0;excluded[i].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();}
        if(i%5==4) excluded[i].velocityInvMass[3]=0;
    }check(excluded,excluded,material);
    for(std::uint32_t generation:{16777216u,2147483648u,UINT32_MAX-1}) {
        auto previous=fixture(2);previous[0].identity[3]=generation;auto current=previous;
        check(previous,current,material);++current[0].identity[3];check(previous,current,material,0,4);
    }
    auto good=fixture(3),bad=good;bad[0].identity[3]++;check(good,bad,material,0,4);
    bad=good;bad[0].positionRadius[3]=0.05f;check(bad,bad,material,0,4);
    bad=good;bad[0].velocityInvMass[3]=-1;check(bad,bad,material,0,4);
    bad=good;bad[0].velocityInvMass[0]=std::numeric_limits<float>::infinity();check(good,bad,material,0,4);
    bad=good;bad[0].positionRadius[0]=std::numeric_limits<float>::quiet_NaN();check(bad,good,material,0,4);
    check(good,bad,material,0,6);
    bad=good;bad[0].positionRadius[0]+=0.5f;check(good,bad,material,0,4);
    bad=good;bad[0].identity[0]=99;check(bad,bad,material,0,6);
    auto invalidMaterial=material;invalidMaterial[0].mechanical[2]=2;check(good,good,invalidMaterial,0,4);
    auto negativeCoefficient=material;negativeCoefficient[0].mechanical[1]=-std::numeric_limits<float>::denorm_min();check(good,good,negativeCoefficient,0,4);
    auto nanCoefficient=material;nanCoefficient[0].mechanical[2]=std::numeric_limits<float>::quiet_NaN();check(good,good,nanCoefficient,0,4);
    auto dead=good;dead[0].identity[0]=UINT32_MAX;dead[0].identity[1]=0;dead[0].identity[3]=0;
    dead[0].positionRadius[0]=dead[0].velocityInvMass[0]=std::numeric_limits<float>::quiet_NaN();check(dead,dead,material);
    check(good,good,material,0,1,4);
    pair[0].velocityInvMass[0]=std::numeric_limits<float>::max();pair[1].velocityInvMass[0]=-std::numeric_limits<float>::max();check(pair,pair,material,0,8);
    pair[0].velocityInvMass[0]=1;pair[1].velocityInvMass[0]=-1;
    for(float mass:{std::numeric_limits<float>::min(),std::numeric_limits<float>::max()}) {pair[0].velocityInvMass[3]=pair[1].velocityInvMass[3]=mass;check(pair,pair,material);}
    GpuContactVelocity recovery(assets,3,0);check(good,good,invalidMaterial,0,4,0,false,&recovery);check(good,good,material,0,0,0,false,&recovery);
    for(float x:{-11.5f,11.5f}) {
        pair=fixture(2);pair[0].positionRadius[0]=x;pair[1].positionRadius[0]=x+0.125f;
        pair[0].velocityInvMass[0]=1;pair[1].velocityInvMass[0]=-1;
        check(pair,pair,makeMaterials(0.5f,0),0,0,0,true,nullptr,false,GranularGridProfile::Basin);
        pair[1].positionRadius[0]=x+0.0625f;
        check(pair,pair,makeMaterials(0.5f,0),0.25f,0,0,false,nullptr,true,GranularGridProfile::Basin);
    }
    std::cout<<"GPU restitution/friction passed: "<<cases<<" fixtures, CPU agreement, dense contacts, closed collision invariants, pipeline chaining and rollback\n";
}
}
