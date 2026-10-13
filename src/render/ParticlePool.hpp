#pragma once
#include "core/Commands.hpp"
#include "core/SpawnPlacement.hpp"
#include "GpuVoxelWorld.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
class ShaderReload;
class GpuSpawnClearance;
// Ordered main-thread submission; all pool membership and allocation stays on GPU.
class ParticlePool {
public:
    ParticlePool(const std::filesystem::path& assets, std::uint32_t capacity);
    ~ParticlePool();
    ParticlePool(const ParticlePool&)=delete;
    ParticlePool& operator=(const ParticlePool&)=delete;
    void rebuild(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,bool resetDiagnostics=false);
    // Optional current terrain rejects non-solid/overlapping/outside spawns
    // before reserving slots. Requires .0625m radius. Lattice placement spaces
    // candidates within one command. Optional clearance rebuilds the frozen
    // particle index before allocation and excludes overlaps with earlier commands.
    // Clearance must have matching capacity and outlive GPU execution.
    void apply(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
               bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const Command& command,float radius=0.045f,const VoxelContactContext* terrain=nullptr,SpawnPlacement placement=SpawnPlacement::Random,GpuSpawnClearance* clearance=nullptr);
    bgfx::DynamicIndexBufferHandle counts() const {return counts_;}
    bgfx::DynamicIndexBufferHandle freeSlots() const {return free_;}
    bgfx::DynamicIndexBufferHandle activeSlots() const {return active_;}
    // Disabled after guarded spawns: partial pool/query reload is not atomic.
    void stageShaders(ShaderReload& reload);
private:
    void release();
    std::uint32_t capacity_;
    bool guardedSpawns_=false;
    bgfx::DynamicIndexBufferHandle counts_=BGFX_INVALID_HANDLE,free_=BGFX_INVALID_HANDLE,active_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,classify_=BGFX_INVALID_HANDLE,spawn_=BGFX_INVALID_HANDLE,voxelSpawn_=BGFX_INVALID_HANDLE,clearSpawn_=BGFX_INVALID_HANDLE,brush_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle pool_=BGFX_INVALID_HANDLE,min_=BGFX_INVALID_HANDLE,max_=BGFX_INVALID_HANDLE,value_=BGFX_INVALID_HANDLE,latticeMin_=BGFX_INVALID_HANDLE,latticeSize_=BGFX_INVALID_HANDLE;
};
}
