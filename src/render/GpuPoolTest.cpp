#include "core/ComputeDispatch.hpp"
#include "GpuPoolTest.hpp"
#include "ParticlePool.hpp"
#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include "assets/Materials.hpp"
#include "core/ParticleTypes.hpp"
#include "../../tests/reference/SweptVoxel.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
namespace pw {
namespace {
template<class T> struct Owned {
    T value;
    explicit Owned(T h):value(h) {if(!bgfx::isValid(h)) throw std::runtime_error("Pool test resource unavailable");}
    ~Owned(){bgfx::destroy(value);}
    Owned(const Owned&)=delete;
};
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
}
void runGpuPoolTest(const std::filesystem::path& assets) {
    Owned reset(loadCompute(assets,"cs_reset"));
    Owned integrate(loadCompute(assets,"cs_particles"));
    Owned exportState(loadCompute(assets,"cs_export"));
    Owned exportPool(loadCompute(assets,"cs_pool_export"));
    Owned sim(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    Owned poolUniform(bgfx::createUniform("u_pool",bgfx::UniformType::Vec4));
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    const auto library=loadMaterials(assets/"materials"/"default.json");const auto packed=library.packed();
    Owned materials(bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),uint32_t(packed.size()*sizeof(MaterialGpu))),lanes,BGFX_BUFFER_COMPUTE_READ));
    for(uint32_t capacity:{1u,127u,128u,129u,255u,256u,257u,16384u}) {
        ParticlePool pool(assets,capacity);
        PoolDiagnostics diagnostics(assets);
        uint64_t snapshotTick=0;
        Owned state(bgfx::createDynamicVertexBuffer(capacity*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Owned next(bgfx::createDynamicVertexBuffer(capacity*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        Owned instances(bgfx::createDynamicVertexBuffer(capacity*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE));
        std::vector<ParticleGpu> initial(capacity),particles(capacity);
        for(uint32_t i=0;i<capacity;++i) initial[i].identity[2]=i;
        Owned upload(bgfx::createDynamicVertexBuffer(bgfx::copy(initial.data(),capacity*sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
        const float simParams[]{0,float(capacity),float(packed.size()),0};
        bgfx::setViewMode(0,bgfx::ViewMode::Sequential);
        bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,upload.value,bgfx::Access::Read);
        bgfx::setBuffer(1,state.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,reset.value,pw::dispatchGroups(capacity));
        pool.rebuild(0,state.value,true);
        const uint16_t rows=uint16_t((capacity+1+255)/256);
        Owned out(bgfx::createTexture2D(256,rows,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned staging(bgfx::createTexture2D(256,rows,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        Owned particleOut(bgfx::createTexture2D(4,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
        Owned particleStaging(bgfx::createTexture2D(4,uint16_t(capacity),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
        std::vector<std::array<uint32_t,4>> membership(uint32_t(rows)*256);
        auto check=[&](uint32_t alive,uint32_t rejected) {
            const float params[]{float(capacity),0,0,0};bgfx::setUniform(poolUniform.value,params);
            bgfx::setBuffer(0,pool.counts(),bgfx::Access::Read);bgfx::setBuffer(1,pool.freeSlots(),bgfx::Access::Read);
            bgfx::setBuffer(2,pool.activeSlots(),bgfx::Access::Read);
            bgfx::setImage(3,out.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exportPool.value,pw::dispatchGroups(capacity+1));
            bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
            bgfx::setImage(1,particleOut.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
            bgfx::dispatch(1,exportState.value,pw::dispatchGroups(capacity));
            bgfx::TextureRegion dst;dst.handle=staging.value;bgfx::TextureRegion src;src.handle=out.value;
            bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,membership.data());
            dst.handle=particleStaging.value;src.handle=particleOut.value;
            bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,particles.data()));
            const auto tag=snapshotTick++;
            // Keep every compute export before the first copy on the pinned Metal backend.
            require(diagnostics.request(1,pool.counts(),tag),"Diagnostic request unexpectedly blocked");
            uint32_t frame;
            do {frame=bgfx::frame();diagnostics.poll(frame);SDL_PumpEvents();} while(frame<ready || diagnostics.pending());
            const auto& snapshot=diagnostics.latest();
            if(!snapshot.valid || snapshot.tick!=tag || snapshot.active!=alive || snapshot.free!=capacity-alive || snapshot.rejected!=rejected)
                throw std::runtime_error("Delayed diagnostic mismatch at capacity "+std::to_string(capacity)+" tick "+std::to_string(tag)+": got "+std::to_string(snapshot.active)+"/"+std::to_string(snapshot.free)+"/"+std::to_string(snapshot.rejected)+" at tick "+std::to_string(snapshot.tick)+", expected "+std::to_string(alive)+"/"+std::to_string(capacity-alive)+"/"+std::to_string(rejected));
            const auto counts=membership[capacity];
            require(counts[0]==alive && counts[1]==capacity-alive && counts[3]==rejected,"Pool count/overflow mismatch");
            std::vector<unsigned> owners(capacity);
            for(unsigned kind=0;kind<2;++kind) {
                const auto count=kind==0?counts[1]:counts[0];
                for(uint32_t i=0;i<count;++i) {
                    const auto slot=membership[i][kind];
                    require(slot<capacity,"Pool slot out of capacity");
                    require(++owners[slot]==1,"Duplicate pool owner");
                    require(bool(particles[slot].identity[1]&1u)==bool(kind),"Pool owner/alive disagreement");
                    require(particles[slot].identity[2]==slot,"Stable slot ID changed");
                }
            }
            for(auto ownersForSlot:owners) require(ownersForSlot==1,"Orphan pool slot");
        };
        check(0,0);
        if(capacity==1) {
            for(unsigned i=0;i<3;++i) require(diagnostics.request(1,pool.counts(),100+i),"Diagnostic ring did not accept a free slot");
            require(!diagnostics.request(1,pool.counts(),999),"Diagnostic ring overwrote a pending destination");
            diagnostics.beginEpoch();
            while(diagnostics.pending()) diagnostics.poll(bgfx::frame());
            require(!diagnostics.latest().valid,"Scene reset adopted a stale snapshot");
            for(unsigned i=0;i<3;++i) require(diagnostics.request(1,pool.counts(),200+i),"Diagnostic ring failed to recycle slots");
            while(diagnostics.pending()) diagnostics.poll(bgfx::frame());
            require(diagnostics.latest().valid && diagnostics.latest().tick==202,"Diagnostic latest-sample ordering failed");
        }
        Command spawn;spawn.type=CommandType::SpawnBrush;spawn.minimum={-1,5,-1};spawn.maximum={1,6,1};
        spawn.parameters={0,0,0,0};spawn.count=1;
        uint64_t sequence=0;
        auto apply=[&](Command command) {command.sequence=sequence++;pool.apply(0,state.value,materials.value,uint32_t(packed.size()),command);};
        // Zero requests are a no-op; then fill exactly to capacity across bounded batches.
        spawn.count=0;apply(spawn);pool.rebuild(0,state.value);check(0,0);
        for(uint32_t remaining=capacity;remaining;) {spawn.count=std::min(remaining,4096u);apply(spawn);remaining-=spawn.count;}
        pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) {
            require(p.identity[3]==1,"Initial generation mismatch");
            require(p.positionRadius[0]>=-1 && p.positionRadius[0]<=1 && p.positionRadius[1]>=5 && p.positionRadius[1]<=6,"Spawn escaped brush");
            require(std::isfinite(p.velocityInvMass[3]) && p.velocityInvMass[3]>0,"Spawn mass invalid");
        }
        Command impulse;impulse.type=CommandType::ApplyImpulse;impulse.minimum={-1,5,-1};impulse.maximum={1,6,1};impulse.parameters={0.25f,1,-0.5f,0};
        apply(impulse);pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) require(p.velocityInvMass[0]==0.25f && p.velocityInvMass[1]==1 && p.velocityInvMass[2]==-0.5f,"Pool impulse mismatch");
        for(unsigned i=1;i<32;++i) apply(impulse);
        // Spatial miss and material miss must preserve the velocity exactly.
        impulse.minimum={2,5,2};impulse.maximum={3,6,3};apply(impulse);
        impulse.minimum={-1,5,-1};impulse.maximum={1,6,1};impulse.material=1;apply(impulse);
        pool.rebuild(0,state.value);check(capacity,0);
        for(const auto& p:particles) require(p.velocityInvMass[0]==8 && p.velocityInvMass[1]==32 && p.velocityInvMass[2]==-16,"Ordered pool impulse/filter mismatch");
        spawn.count=127;apply(spawn);pool.rebuild(0,state.value);check(capacity,127);
        Command remove;remove.type=CommandType::RemoveBrush;remove.minimum={-8,0,-8};remove.maximum={8,12,8};
        apply(remove);apply(remove);pool.rebuild(0,state.value);check(0,127);
        spawn.count=std::min(capacity,128u);spawn.parameters[3]=1.0f/60.0f;apply(spawn);
        pool.rebuild(0,state.value);check(spawn.count,127);
        for(const auto& p:particles) if(p.identity[1]&1u) require(p.identity[3]==2,"Reused slot generation did not advance");
        // Expiry uses the regular integration path and masks the fixed-capacity render record.
        const float step[]{1.0f/60.0f,float(capacity),float(packed.size()),0};
        bgfx::setUniform(sim.value,step);bgfx::setBuffer(0,state.value,bgfx::Access::Read);
        bgfx::setBuffer(1,next.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
        bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,integrate.value,pw::dispatchGroups(capacity));
        std::swap(state.value,next.value);pool.rebuild(0,state.value);check(0,127);
        // Export the first two lanes of each texture row to verify dead render records.
        const float renderParams[]{0,float(capacity),0,2};
        bgfx::setUniform(sim.value,renderParams);bgfx::setBuffer(0,instances.value,bgfx::Access::Read);
        bgfx::setImage(1,particleOut.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(1,exportState.value,pw::dispatchGroups(capacity));
        bgfx::TextureRegion renderDst;renderDst.handle=particleStaging.value;
        bgfx::TextureRegion renderSrc;renderSrc.handle=particleOut.value;
        bgfx::blit(2,renderDst,renderSrc);
        std::vector<ParticleGpu> rendered(capacity);
        const auto renderReady=bgfx::read(renderDst,rendered.data());
        while(bgfx::frame()<renderReady) SDL_PumpEvents();
        for(const auto& record:rendered) for(unsigned field=0;field<4;++field)
            require(record.positionRadius[field]==0 && record.velocityInvMass[field]==0,"Dead instance was not masked");
        // Terrain-aware allocation rejects before reserving a physical slot.
        // This resident region sits outside the legacy x/z +/-8m grid.
        terrain::SparseVoxelStore terrainStore(terrain::ResidentChunkLayout({-3,0,-2},{2,2,2}),uint32_t(packed.size()));
        const auto occupied=*terrain::addressOf({-92,48,-60});
        terrainStore.setMaterial(occupied,0,terrainStore.revision());
        GpuVoxelWorld world(terrainStore,{});VoxelContactContext terrainContext{world,terrainStore,{}};
        pool.rebuild(0,state.value,true);check(0,0);
        const auto beforeRejected=particles;
        Command terrainSpawn;terrainSpawn.type=CommandType::SpawnBrush;terrainSpawn.count=std::min(capacity,257u);
        terrainSpawn.minimum=terrainSpawn.maximum={-11.4375f,6.0625f,-7.4375f};
        auto terrainApply=[&](const Command& command){pool.apply(0,state.value,materials.value,uint32_t(packed.size()),command,.0625f,&terrainContext);};
        terrainApply(terrainSpawn);pool.rebuild(0,state.value);check(0,terrainSpawn.count);
        terrainSpawn.minimum=terrainSpawn.maximum={-12,6,-7};
        terrainApply(terrainSpawn);pool.rebuild(0,state.value);check(0,2*terrainSpawn.count);
        // Liquid phases cannot enter the current granular terrain pool.
        terrainSpawn.minimum=terrainSpawn.maximum={-10.5f,6,-6.5f};terrainSpawn.material=2;
        terrainApply(terrainSpawn);pool.rebuild(0,state.value);check(0,3*terrainSpawn.count);
        require(!std::memcmp(beforeRejected.data(),particles.data(),capacity*sizeof(ParticleGpu)),"Rejected terrain spawn mutated slot state/generation");
        // Exact face tangency is admissible and prior rejections used no capacity.
        terrainSpawn.material=0;terrainSpawn.minimum=terrainSpawn.maximum={-11.4375f,6.1875f,-7.4375f};
        terrainApply(terrainSpawn);pool.rebuild(0,state.value);check(terrainSpawn.count,3*terrainSpawn.count);
        for(unsigned slot=0;slot<capacity;++slot)if(particles[slot].identity[1]&1u) {
            require(particles[slot].positionRadius[3]==.0625f && particles[slot].positionRadius[1]==6.1875f,"Terrain spawn changed radius or contact point");
            require(particles[slot].identity[3]==beforeRejected[slot].identity[3]+1,"Terrain spawn did not advance selected slot generation exactly once");
        }
        const auto beforeStale=particles;
        auto rejectedContext=[&](const VoxelContactContext& context,float radius=.0625f) {
            bool caught=false;try{pool.apply(0,state.value,materials.value,uint32_t(packed.size()),terrainSpawn,radius,&context);}catch(const std::invalid_argument&){caught=true;}
            require(caught,"Pool accepted invalid terrain context/radius");
        };
        rejectedContext(terrainContext,.045f);
        VoxelContactContext wrongOrigin{world,terrainStore,{1,0,0}};rejectedContext(wrongOrigin);
        terrain::SparseVoxelStore foreign(terrain::ResidentChunkLayout({-3,0,-2},{2,2,2}),uint32_t(packed.size()));
        VoxelContactContext wrongOwner{world,foreign,{}};rejectedContext(wrongOwner);
        terrainStore.setMaterial(occupied,terrain::SparseVoxelStore::empty,terrainStore.revision());rejectedContext(terrainContext);
        pool.rebuild(0,state.value);check(terrainSpawn.count,3*terrainSpawn.count);
        require(!std::memcmp(beforeStale.data(),particles.data(),capacity*sizeof(ParticleGpu)),"Rejected pool context mutated particle state");
        GpuVoxelWorld editedWorld(terrainStore,{});VoxelContactContext edited{editedWorld,terrainStore,{}};
        Command clearTerrain;clearTerrain.type=CommandType::RemoveBrush;clearTerrain.minimum={-12,0,-8};clearTerrain.maximum={-4,8,0};
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),clearTerrain,.0625f,&edited);pool.rebuild(0,state.value,true);check(0,0);
        terrainSpawn.minimum=terrainSpawn.maximum={-11.4375f,6.0625f,-7.4375f};
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),terrainSpawn,.0625f,&edited);pool.rebuild(0,state.value);check(terrainSpawn.count,0);
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),clearTerrain,.0625f,&edited);pool.rebuild(0,state.value,true);check(0,0);
        // A single mixed brush must not spend free ranks on blocked requests.
        terrainStore.setMaterial(occupied,0,terrainStore.revision());GpuVoxelWorld mixedWorld(terrainStore,{});VoxelContactContext mixedContext{mixedWorld,terrainStore,{}};
        Command mixedSpawn=terrainSpawn;mixedSpawn.count=257;mixedSpawn.sequence=29;
        mixedSpawn.minimum={-11.49f,6.0625f,-7.4375f};mixedSpawn.maximum={-11.2f,6.0625f,-7.4375f};
        auto hash=[](uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;};
        unsigned clearRequests=0;const auto voxelLower=*terrain::relativeCorner(occupied,{});
        for(unsigned request=0;request<257;++request) {
            const float random=float(hash(29u^hash(request))>>8)/16777216.0f;
            const double x=std::lerp(double(mixedSpawn.minimum[0]),double(mixedSpawn.maximum[0]),double(random));
            clearRequests+=reference::voxelDistance({x,6.0625,-7.4375},voxelLower)>=.0625;
        }
        require(clearRequests>0 && clearRequests<257,"Mixed terrain fixture is not mixed");
        const auto accepted=std::min(capacity,clearRequests);
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),mixedSpawn,.0625f,&mixedContext);pool.rebuild(0,state.value);check(accepted,257-accepted);
        for(const auto& particle:particles)if(particle.identity[1]&1u)
            require(reference::voxelDistance({particle.positionRadius[0],particle.positionRadius[1],particle.positionRadius[2]},voxelLower)>=.0625,"Mixed brush spawned inside terrain");
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),clearTerrain,.0625f,&mixedContext);pool.rebuild(0,state.value,true);check(0,0);
        // Distinct lattice sites provide within-command particle spacing.
        Command latticeSpawn=terrainSpawn;latticeSpawn.sequence=77;
        for(unsigned requested:{17u,257u}) {
        latticeSpawn.count=requested;
        latticeSpawn.minimum={-11.49f,6.0f,-7.49f};latticeSpawn.maximum={-10.51f,6.25f,-6.51f};
        const auto lattice=spawnLattice(latticeSpawn.minimum,latticeSpawn.maximum);
        std::set<std::array<float,3>> admissible;
        unsigned latticeRejected=0;
        for(unsigned request=0;request<latticeSpawn.count;++request) {
            const auto position=spawnLatticePosition(lattice,request,uint32_t(latticeSpawn.sequence));
            if(position && reference::voxelDistance({(*position)[0],(*position)[1],(*position)[2]},voxelLower)>=.0625)admissible.insert(*position);
            else ++latticeRejected;
        }
        const auto latticeAccepted=std::min(capacity,uint32_t(admissible.size()));
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),latticeSpawn,.0625f,&mixedContext,SpawnPlacement::GranularLattice);
        pool.rebuild(0,state.value);check(latticeAccepted,latticeSpawn.count-latticeAccepted);
        std::set<std::array<float,3>> spawned;
        for(const auto& particle:particles)if(particle.identity[1]&1u) {
            const std::array<float,3> position{particle.positionRadius[0],particle.positionRadius[1],particle.positionRadius[2]};
            require(admissible.contains(position) && spawned.insert(position).second,"Lattice GPU spawn repeated or escaped an admissible site");
        }
        if(capacity>=admissible.size())require(spawned==admissible,"Lattice GPU spawn lost valid sites");
        for(auto i=spawned.begin();i!=spawned.end();++i)for(auto j=spawned.begin();j!=i;++j) {
            double squared=0;for(unsigned axis=0;axis<3;++axis){double d=double((*i)[axis])-(*j)[axis];squared+=d*d;}
            require(squared>=.015625,"Lattice brush created overlapping grains");
        }
        if(requested==257)require(latticeRejected>0,"Lattice fixture did not exercise candidate rejection");
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),clearTerrain,.0625f,&mixedContext);pool.rebuild(0,state.value,true);check(0,0);
        }
        latticeSpawn.minimum={-10.49f,6.01f,-6.49f};latticeSpawn.maximum={-10.4f,6.1f,-6.4f};
        pool.apply(0,state.value,materials.value,uint32_t(packed.size()),latticeSpawn,.0625f,&mixedContext,SpawnPlacement::GranularLattice);
        pool.rebuild(0,state.value);check(0,latticeSpawn.count);
        pool.rebuild(0,state.value,true);check(0,0);
        if(capacity==1) {
            // Exhausted generations must never wrap to a previously valid handle.
            particles[0].identity[3]=0xffffffffu;
            Owned exhausted(bgfx::createDynamicVertexBuffer(bgfx::copy(particles.data(),sizeof(ParticleGpu)),lanes,BGFX_BUFFER_COMPUTE_READ));
            bgfx::setUniform(sim.value,simParams);bgfx::setBuffer(0,exhausted.value,bgfx::Access::Read);
            bgfx::setBuffer(1,state.value,bgfx::Access::Write);bgfx::setBuffer(2,instances.value,bgfx::Access::Write);
            bgfx::setBuffer(3,materials.value,bgfx::Access::Read);bgfx::dispatch(0,reset.value,1);
            pool.rebuild(0,state.value,true);spawn.count=1;apply(spawn);pool.rebuild(0,state.value);check(0,1);
            require(particles[0].identity[3]==0xffffffffu,"Slot generation wrapped");
        }
        require(diagnostics.request(1,pool.counts(),snapshotTick++),"Shutdown diagnostic request failed");
        std::cout<<"GPU pool ownership/full/reuse/lifetime/voxel-spawn/delayed diagnostics passed: "<<capacity<<" slots\n";
    }
}
}
