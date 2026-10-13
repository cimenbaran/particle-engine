#include "GpuDebrisReservation.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
#include <SDL3/SDL.h>
#include <cstring>
#include <iostream>
namespace pw {
namespace {
template<class T>struct ReservationHandle {
    T value;explicit ReservationHandle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Reservation test allocation failed");}
    ~ReservationHandle(){bgfx::destroy(value);}ReservationHandle(const ReservationHandle&)=delete;
};
}
void runGpuDebrisReservationTest(const std::filesystem::path& assets) {
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    ReservationHandle exporter(loadCompute(assets,"cs_export"));
    ReservationHandle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    unsigned cases=0;std::uint64_t checkedSlots=0;
    for(std::uint32_t capacity:{0u,1u,127u,128u,129u,255u,256u,257u,4096u,16384u}) {
        const auto storage=std::max(1u,capacity);GpuDebrisReservation reservation(assets,capacity);PoolDiagnostics diagnostic(assets);
        ReservationHandle source(bgfx::createDynamicVertexBuffer(storage*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE));
        ReservationHandle image(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        ReservationHandle staging(bgfx::createTexture2D(4,std::uint16_t(storage),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::uint32_t frame=bgfx::frame();
        auto read=[&](bgfx::DynamicVertexBufferHandle state,unsigned lanes) {
            const float params[]{0,float(capacity),0,float(lanes)};
            bgfx::setUniform(uniform.value,params);bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setImage(1,image.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            if(capacity)bgfx::dispatch(5,exporter.value,dispatchGroups(capacity));
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(7,dst,src);
            std::vector<ParticleGpu> result(storage);const auto ready=bgfx::read(dst,result.data());
            while((frame=bgfx::frame())<ready)SDL_PumpEvents();diagnostic.poll(frame);return result;
        };
        for(unsigned pattern=0;pattern<6;++pattern) {
            std::vector<ParticleGpu> input(storage);
            for(unsigned i=0;i<capacity;++i) {
                auto& p=input[i];p.positionRadius[0]=float(i)*.01f;p.positionRadius[1]=2;p.positionRadius[3]=.0625f;
                p.velocityInvMass[0]=.5f;p.velocityInvMass[3]=1;p.thermalState[0]=293.15f;p.thermalState[3]=3;
                p.identity[0]=1;p.identity[1]=pattern==2 || (pattern==1 && i%3==0)?1:0;p.identity[2]=i;p.identity[3]=i+1;
                if(pattern==0 && i<2)p.identity[3]=i==0?0u:UINT32_MAX-1;
                if(pattern==1 && i%11==0 && !p.identity[1])p.identity[3]=UINT32_MAX;
            }
            if(capacity) {
                if(pattern==3)input[capacity-1].identity[1]=2; // pending snapshot is not a live source
                if(pattern==4){input[capacity-1].identity[1]=1;input[capacity-1].identity[3]=0;}
                if(pattern==5){input[capacity-1].identity[1]=1;input[capacity-1].identity[2]=capacity;}
            }
            bgfx::update(source.value,0,bgfx::copy(input.data(),storage*sizeof(ParticleGpu)));
            std::vector<unsigned> eligible;unsigned invalid=0;
            for(unsigned i=0;i<capacity;++i) {
                const auto& id=input[i].identity;const bool valid=id[1]<=1 && (id[1]==0 || (id[2]==i && id[3]!=0));
                if(!valid)invalid=2;
                if(valid && id[1]==0 && id[3]!=UINT32_MAX)eligible.push_back(i);
            }
            const auto available=std::uint32_t(eligible.size());
            for(auto requested:{available,0u,available+1,available/2,1u,UINT32_MAX}) {
                reservation.submit(1,source.value,requested);require(diagnostic.request(6,reservation.status(),cases),"Reservation diagnostic request failed");
                const auto actual=read(reservation.snapshot(),4);const auto claims=read(reservation.claims(),1);const auto unchanged=read(source.value,4);
                const auto errors=invalid|(requested>available?1u:0u),reserved=errors?0:requested;
                const auto& stats=diagnostic.latest();
                require(stats.valid && stats.active==errors && stats.free==requested && stats.reserved==available && stats.rejected==reserved,"Reservation status disagrees with CPU allocator");
                auto expected=input;
                for(unsigned rank=0;rank<reserved;++rank){auto slot=eligible[rank];auto& id=expected[slot].identity;id[1]=2;id[2]=slot;++id[3];}
                for(unsigned i=0;i<capacity;++i) {
                    require(std::memcmp(&actual[i],&expected[i],64)==0,"Reservation snapshot mismatch or partial allocation");
                    require(std::memcmp(&unchanged[i],&input[i],64)==0,"Reservation mutated live input");
                    std::array<std::uint32_t,4> words{};std::memcpy(words.data(),&claims[i],16);
                    const auto expectedClaim=i<reserved?std::array<std::uint32_t,4>{eligible[i],input[eligible[i]].identity[3]+1,input[eligible[i]].identity[3],0}:std::array<std::uint32_t,4>{UINT32_MAX,0,0,0};
                    require(words==expectedClaim,"Reservation rank, generation or cleared tail mismatch");
                }
                ++cases;checkedSlots+=capacity;
            }
        }
        for(auto alias:{reservation.snapshot(),reservation.claims(),bgfx::DynamicVertexBufferHandle{bgfx::kInvalidHandle}}) {
            bool rejected=false;try{reservation.submit(1,alias,1);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Reservation accepted invalid/aliased input");
        }
    }
    std::cout<<"GPU debris reservation passed: "<<cases<<" cases, "<<checkedSlots<<" slot checks; deterministic all-or-nothing claims, full pools, generation exhaustion, invalid identities, reuse and immutable input\n";
}
}
