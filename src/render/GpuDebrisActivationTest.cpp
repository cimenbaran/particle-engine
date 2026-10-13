#include "GpuDebrisActivation.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/TerrainMass.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
#include <limits>
namespace pw {
namespace {
template<class T>struct ActivationHandle {
    T value;explicit ActivationHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Activation test allocation failed");}
    ~ActivationHandle(){bgfx::destroy(value);}ActivationHandle(const ActivationHandle&)=delete;
};
}
void runGpuDebrisActivationTest(const std::filesystem::path& assets) {
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    const auto materials=loadMaterials(assets/"materials/default.json");
    ActivationHandle exporter(loadCompute(assets,"cs_export"));ActivationHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    auto makeSeeds=[](unsigned n) {std::vector<DebrisSeed> seeds(n);for(unsigned i=0;i<n;++i){auto& s=seeds[i];s.position={-1+float(i%8)*.125f,2+float((i/8)%8)*.125f,-1+float(i/64)*.125f};s.velocity={1,-2,3};s.material=i%3==2?3:i%3;s.temperature=293.15f+float(i)*.125f;s.fuel=s.material==3?.5f:0;}return seeds;};
    unsigned cases=0;
    for(unsigned capacity:{0u,1u,127u,128u,129u,255u,256u,257u,4096u}) {
        const auto storage=std::max(1u,capacity);
        ActivationHandle source(bgfx::createDynamicVertexBuffer(storage*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE));
        ActivationHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ActivationHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        PoolDiagnostics diagnostics(assets);std::uint32_t frame=bgfx::frame();
        auto read=[&](bgfx::DynamicVertexBufferHandle handle) {
            const float p[]{0,float(capacity),0,4};bgfx::setUniform(uniform.value,p);bgfx::setBuffer(0,handle,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            if(capacity)bgfx::dispatch(5,exporter.value,dispatchGroups(capacity));
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(7,dst,src);std::vector<ParticleGpu> result(storage);
            const auto ready=bgfx::read(dst,result.data());while((frame=bgfx::frame())<ready)SDL_PumpEvents();diagnostics.poll(frame);return result;
        };
        for(unsigned pattern=0;pattern<4;++pattern) {
            std::vector<ParticleGpu> input(storage);std::vector<unsigned> free;
            for(unsigned i=0;i<capacity;++i) {
                auto& p=input[i];p.positionRadius[0]=p.positionRadius[1]=p.positionRadius[2]=10;p.positionRadius[3]=.0625f;p.velocityInvMass[3]=1;p.thermalState[0]=293.15f;p.thermalState[3]=5;
                p.identity[0]=1;p.identity[1]=pattern==2 || (pattern==1 && i%3==0)?1:0;p.identity[2]=i;p.identity[3]=10+i;
                if(pattern==3 && i==0)p.identity[3]=UINT32_MAX;
                if(!p.identity[1] && p.identity[3]!=UINT32_MAX)free.push_back(i);
            }
            const unsigned available=unsigned(free.size());
            for(unsigned wanted:{0u,std::min(17u,available),std::min(4096u,available),std::min(4096u,available+1)}) {
                bgfx::update(source.value,0,bgfx::copy(input.data(),storage*sizeof(ParticleGpu)));
                GpuDebrisReservation reservation(assets,capacity);const auto seeds=makeSeeds(wanted);GpuDebrisActivation activation(assets,capacity,materials,seeds);
                bool rejected=false;try{activation.submit(1,reservation);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Unsubmitted reservation accepted");
                reservation.submit(0,source.value,wanted);
                if(capacity==1 && pattern==0 && wanted==0) {
                    GpuDebrisActivation wrongCapacity(assets,2,materials,seeds);bool mismatch=false;
                    try{wrongCapacity.submit(1,reservation);}catch(const std::invalid_argument&){mismatch=true;}require(mismatch,"Mismatched reservation capacity accepted");
                }
                activation.submit(1,reservation);diagnostics.request(6,activation.status(),cases);
                const auto actual=read(activation.output()),reserved=read(reservation.snapshot()),unchanged=read(source.value);
                const bool success=wanted<=available;const auto& status=diagnostics.latest();
                require(status.valid && (success?status.active==0:(status.active&1)!=0) && status.free==wanted && status.rejected==(success?wanted:0),"Activation status mismatch");
                auto expected=input,expectedReservation=input;
                if(success)for(unsigned rank=0;rank<wanted;++rank) {
                    const auto slot=free[rank];const auto& seed=seeds[rank];auto& p=expected[slot];p={};
                    for(unsigned a=0;a<3;++a){p.positionRadius[a]=seed.position[a];p.velocityInvMass[a]=seed.velocity[a];}
                    p.positionRadius[3]=.0625f;p.velocityInvMass[3]=terrain::conversionMassProfile(materials.entries[seed.material].gpu.mechanical[0]).inverseMass;
                    p.thermalState[0]=seed.temperature;p.thermalState[1]=seed.fuel;p.identity[0]=seed.material;p.identity[1]=1;p.identity[2]=slot;p.identity[3]=input[slot].identity[3]+1;
                    expectedReservation[slot].identity[1]=2;expectedReservation[slot].identity[3]=p.identity[3];
                }
                for(unsigned i=0;i<capacity;++i){require(std::memcmp(&actual[i],&expected[i],64)==0,"Activated particle fields or unchanged slots differ");require(std::memcmp(&unchanged[i],&input[i],64)==0,"Activation mutated live state");require(std::memcmp(&reserved[i],&expectedReservation[i],64)==0,"Activation mutated its reservation input");}
                rejected=false;try{activation.submit(1,reservation);}catch(const std::logic_error&){rejected=true;}require(rejected,"Activation accepted repeated submission");
                require(activation.materialHash()==materials.contentHash,"Activation lost material epoch");++cases;
            }
        }
        if(capacity>=2) {
            for(unsigned fault=0;fault<9;++fault) {
                std::vector<ParticleGpu> input(storage);for(unsigned i=0;i<capacity;++i){input[i].identity[2]=i;input[i].identity[3]=1;}
                bgfx::update(source.value,0,bgfx::copy(input.data(),storage*sizeof(ParticleGpu)));
                GpuDebrisReservation reservation(assets,capacity);reservation.submit(0,source.value,2);auto reserved=read(reservation.snapshot());
                // Inject into the private preparation only, after its GPU pass.
                if(fault<4) {
                    std::array<std::uint32_t,4> claim{0,2,1,0};
                    if(fault==0)claim[0]=capacity; // bounds
                    if(fault==1)claim[1]=3; // generation relation
                    if(fault==2)claim[3]=1; // malformed record
                    const unsigned rank=fault==3?1:0; // duplicate slot
                    bgfx::update(reservation.claims(),rank,bgfx::copy(claim.data(),16));
                } else if(fault==4) {
                    reserved[0].identity[3]=3;bgfx::update(reservation.snapshot(),3,bgfx::copy(reserved[0].identity,16));
                }
                if(fault==6) {
                    const std::array<std::uint32_t,8> reversed{1,2,1,0,0,2,1,0};bgfx::update(reservation.claims(),0,bgfx::copy(reversed.data(),32));
                }
                if(fault==7 && capacity>2) {
                    reserved[2].identity[1]=2;reserved[2].identity[3]=2;bgfx::update(reservation.snapshot(),11,bgfx::copy(reserved[2].identity,16));
                }
                if(fault==8) {
                    reserved[0].identity[1]=0;bgfx::update(reservation.snapshot(),3,bgfx::copy(reserved[0].identity,16));
                }
                const auto seeds=makeSeeds(fault==5?1:2);GpuDebrisActivation activation(assets,capacity,materials,seeds);
                activation.submit(1,reservation);diagnostics.request(6,activation.status(),cases);const auto actual=read(activation.output());
                require(diagnostics.latest().valid && diagnostics.latest().active!=0 && diagnostics.latest().rejected==0,"Bad reservation partially activated");
                for(unsigned i=0;i<capacity;++i)require(std::memcmp(&actual[i],&reserved[i],64)==0,"Failed activation did not preserve full private snapshot");
                ++cases;
            }
        }
    }
    auto rejects=[&](std::vector<DebrisSeed> seeds){bool bad=false;try{GpuDebrisActivation activation(assets,1,materials,seeds);}catch(const std::invalid_argument&){bad=true;}require(bad,"Invalid debris seed accepted");};
    auto valid=makeSeeds(1);
    for(unsigned bad=0;bad<11;++bad) {
        auto seeds=valid;
        if(bad==0)seeds[0].material=2;
        if(bad==1)seeds[0].material=UINT32_MAX;
        if(bad==2)seeds[0].position[0]=16;
        if(bad==3)seeds[0].position[1]=0;
        if(bad==4)seeds[0].velocity={8,8,8};
        if(bad==5)seeds[0].temperature=-1;
        if(bad==6)seeds[0].fuel=2;
        if(bad==7)seeds[0].position[2]=std::numeric_limits<float>::quiet_NaN();
        if(bad==8)seeds[0].temperature=std::numeric_limits<float>::quiet_NaN();
        if(bad==9)seeds[0].fuel=std::numeric_limits<float>::infinity();
        if(bad==10)seeds[0].velocity[1]=std::numeric_limits<float>::infinity();
        rejects(seeds);
    }
    rejects(makeSeeds(4097));
    std::cout<<"GPU debris activation passed: "<<cases<<" batches, complete particle initialization, encoded masses, immutable live/reserved state, malformed claims, generation mismatch, count failure and host seed rejection\n";
}
}
