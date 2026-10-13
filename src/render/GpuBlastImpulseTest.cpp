#include "GpuBlastImpulse.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
namespace pw {
namespace {
template<class T>struct BlastHandle {
    T value;explicit BlastHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Blast test allocation failed");}
    ~BlastHandle(){bgfx::destroy(value);}BlastHandle(const BlastHandle&)=delete;
};
}
void runGpuBlastImpulseTest(const std::filesystem::path& assets) {
    const auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    const auto rejects=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad,"Invalid blast impulse input accepted");};
    const auto library=loadMaterials(assets/"materials/default.json");const auto packed=library.packed();
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    BlastHandle materials(bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),unsigned(packed.size()*64)),lanes,BGFX_BUFFER_COMPUTE_READ));
    BlastHandle exporter(loadCompute(assets,"cs_blast_export"));BlastHandle uniform(bgfx::createUniform("u_blast_export",bgfx::UniformType::Vec4));
    BlastImpulseRequest request;request.centre={0,4,0};request.radius=1;request.speed=2;request.joules=8;
    unsigned cases=0;std::uint64_t checked=0;
    const auto fixture=[](unsigned n){std::vector<ParticleGpu> result(n);for(unsigned i=0;i<n;++i)result[i]={{.25f,4,0,.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,i,1}};return result;};
    const auto test=[&](std::vector<ParticleGpu> input,BlastImpulseRequest r,unsigned expectedError=0,int expectedAccepted=-1) {
        const auto count=unsigned(input.size());auto upload=input;if(upload.empty())upload.resize(1);
        BlastHandle source(bgfx::createDynamicVertexBuffer(bgfx::copy(upload.data(),unsigned(upload.size()*64)),lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        const auto height=std::uint16_t((std::max(1u,count)+255u)/256u);
        BlastHandle image(bgfx::createTexture2D(1024,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        BlastHandle staging(bgfx::createTexture2D(1024,height,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        const auto read=[&](bgfx::DynamicVertexBufferHandle state) {
            std::vector<ParticleGpu> result(std::size_t(height)*256);
            const float p[]{float(count),0,0,0};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(2,exporter.value,dispatchGroups(std::max(1u,count)));
            bgfx::TextureRegion a,b;a.handle=image.value;b.handle=staging.value;bgfx::blit(3,b,a);const auto ready=bgfx::read(b,result.data());while(bgfx::frame()<ready)SDL_PumpEvents();result.resize(count);return result;
        };
        GpuBlastImpulse impulse(assets,count,r);
        require(impulse.joulesPerTicket()*impulse.ticketBudget()<=r.joules,"Encoded ticket budget exceeds input energy");
        rejects([&]{impulse.submit(0,impulse.output(),materials.value,unsigned(packed.size()));});
        rejects([&]{impulse.submit(0,source.value,materials.value,0);});
        impulse.submit(0,source.value,materials.value,unsigned(packed.size()));
        bool repeated=false;try{impulse.submit(0,source.value,materials.value,unsigned(packed.size()));}catch(const std::logic_error&){repeated=true;}require(repeated,"Repeated blast impulse submission accepted");
        PoolDiagnostics diagnostics(assets);diagnostics.request(1,impulse.status(),0);const auto actual=read(impulse.output());const auto original=read(source.value);diagnostics.poll(bgfx::frame());const auto status=diagnostics.latest();
        require(status.valid && status.active==expectedError,"Wrong blast impulse error flags");
        require(!count || std::memcmp(input.data(),original.data(),count*64)==0,"Blast impulse changed source state");
        if(expectedError){require(status.reserved==0 && status.rejected==0 && std::memcmp(input.data(),actual.data(),count*64)==0,"Faulted impulse changed private state or charged energy");}
        else {
            unsigned changed=0;double positiveWork=0;
            for(unsigned i=0;i<count;++i) {
                auto untouched=input[i];for(unsigned a=0;a<3;++a)untouched.velocityInvMass[a]=actual[i].velocityInvMass[a];
                require(std::memcmp(&untouched,&actual[i],64)==0,"Blast impulse changed non-velocity lanes");
                const bool different=std::memcmp(input[i].velocityInvMass,actual[i].velocityInvMass,12)!=0;
                if(!different)continue;++changed;
                require(input[i].identity[1]==1 && packed[input[i].identity[0]].identity[1]==0 && input[i].velocityInvMass[3]>0,"Blast moved dead/static/nonsolid particle");
                double before=0,after=0;for(unsigned a=0;a<3;++a){require(std::isfinite(actual[i].velocityInvMass[a]),"Nonfinite impulse output");before+=double(input[i].velocityInvMass[a])*input[i].velocityInvMass[a];after+=double(actual[i].velocityInvMass[a])*actual[i].velocityInvMass[a];}
                require(after<=144,"Blast impulse exceeded 12 m/s");positiveWork+=std::max(0.,.5*(after-before)/input[i].velocityInvMass[3]);
                std::array<double,3> offset;for(unsigned a=0;a<3;++a)offset[a]=double(input[i].positionRadius[a])-r.centre[a];const double distance=std::hypot(offset[0],offset[1],offset[2]);
                require(distance<r.radius+1e-6,"Blast moved an out-of-radius particle");
                const double falloff=std::max(0.,1-distance/r.radius);std::array<double,3> expected;
                for(unsigned a=0;a<3;++a)expected[a]=input[i].velocityInvMass[a]+(distance<1e-6?(a==1?1.:0.):offset[a]/distance)*r.speed*falloff*falloff;
                const double speed=std::hypot(expected[0],expected[1],expected[2]);if(speed>11.99998)for(auto& v:expected)v*=11.99998/speed;
                for(unsigned a=0;a<3;++a)require(std::abs(expected[a]-actual[i].velocityInvMass[a])<.00005,"Radial impulse differs from double reference");
            }
            require(changed==status.reserved && status.free>=changed && status.free<=count,"Impulse counts disagree with output");
            const double charged=double(status.rejected)*impulse.joulesPerTicket();
            require(status.rejected<=impulse.ticketBudget() && positiveWork<=charged && charged<=r.joules,"Impulse exceeds positive kinetic-work budget");
            if(expectedAccepted>=0){require(changed==unsigned(expectedAccepted),"Wrong deterministic energy prefix");for(unsigned i=0;i<count;++i)require((std::memcmp(input[i].velocityInvMass,actual[i].velocityInvMass,12)!=0)==(i<changed),"Energy allocation wrapped or lost slot order");}
        }
        ++cases;checked+=count;
    };
    for(unsigned count:{0u,1u,127u,128u,129u,255u,256u,257u,1024u,4096u,65537u,1048576u})test(fixture(count),request,0,int(std::min(12u,count)));
    auto r=request;r.joules=0;test(fixture(1024),r,0,0);r.joules=1e-6;test(fixture(1024),r,0,0);r.joules=1e9;test(fixture(1024),r,0,1024);
    r=request;r.speed=0;test(fixture(1024),r,0,0);
    std::mt19937 random(40019);std::uniform_real_distribution<float> unit(-1,1);
    for(unsigned trial=0;trial<208;++trial) {
        auto input=fixture(2048);r=request;r.joules=std::pow(10.,int(trial%16)-6);r.speed=float(trial%13);
        for(unsigned i=0;i<input.size();++i) {
            auto& p=input[i];for(unsigned a=0;a<3;++a){p.positionRadius[a]=r.centre[a]+unit(random)*1.5f;p.velocityInvMass[a]=unit(random)*6;}
            p.velocityInvMass[3]=std::pow(10.f,float(int(random()%13)-6));
            if(i%11==0)p.identity[1]=0;if(i%13==0)p.velocityInvMass[3]=0;if(i%17==0)p.identity[0]=2;
        }
        test(std::move(input),r);
    }
    {
        auto input=fixture(12);r=request;r.joules=1e9;r.speed=12;
        input[0].positionRadius[0]=0;input[0].velocityInvMass[0]=12; // centre fallback + speed cap
        input[1].positionRadius[0]=1; // exact boundary
        input[2].positionRadius[0]=std::nextafter(1.f,0.f);
        input[3].positionRadius[0]=std::nextafter(1.f,2.f);
        input[4].positionRadius[0]=.0000005f; // deterministic near-centre direction
        input[5].velocityInvMass[0]=-12; // braking must not credit other grains
        input[6].velocityInvMass[3]=.000001f;input[7].velocityInvMass[3]=1000000;
        input[8].velocityInvMass[3]=0;input[9].identity[1]=0;input[10].identity[0]=2;input[11].identity[3]=UINT32_MAX;
        test(std::move(input),r);
    }
    for(unsigned fault=0;fault<10;++fault) {
        auto input=fixture(256);auto& p=input[200];unsigned error=2;
        if(fault==0)p.positionRadius[0]=INFINITY;if(fault==1)p.velocityInvMass[1]=NAN;if(fault==2)p.velocityInvMass[3]=-1;if(fault==3)p.positionRadius[3]=0;if(fault==4)p.velocityInvMass[0]=13;if(fault==5)p.velocityInvMass[3]=1e7f;
        if(fault==6){p.identity[2]=1;error=1;}if(fault==7){p.identity[1]=2;error=1;}if(fault==8){p.identity[3]=0;error=1;}if(fault==9){p.identity[0]=100;error=4;}
        test(std::move(input),request,error);
    }
    for(unsigned fault=0;fault<9;++fault){auto bad=request;unsigned count=1;if(fault==0)count=1048577;if(fault==1)bad.radius=0;if(fault==2)bad.radius=5;if(fault==3)bad.speed=13;if(fault==4)bad.joules=-1;if(fault==5)bad.joules=1e-9;if(fault==6)bad.joules=1e9+1;if(fault==7)bad.centre[0]=NAN;if(fault==8)bad.centre[1]=-1;rejects([&]{GpuBlastImpulse invalid(assets,count,bad);});}
    std::cout<<"GPU blast impulse passed: "<<cases<<" batches / "<<checked<<" slots; radial reference, positive-work energy bounds, saturating allocation, immutable source and fault rollback\n";
}
}
