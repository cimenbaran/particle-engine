#pragma once
#include "core/Commands.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
class ShaderReload;
// Ordered main-thread submission; all pool membership and allocation stays on GPU.
class ParticlePool {
public:
    ParticlePool(const std::filesystem::path& assets, std::uint32_t capacity);
    ~ParticlePool();
    ParticlePool(const ParticlePool&)=delete;
    ParticlePool& operator=(const ParticlePool&)=delete;
    void rebuild(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,bool resetDiagnostics=false);
    void apply(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
               bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const Command& command);
    bgfx::DynamicIndexBufferHandle counts() const {return counts_;}
    bgfx::DynamicIndexBufferHandle freeSlots() const {return free_;}
    bgfx::DynamicIndexBufferHandle activeSlots() const {return active_;}
    void stageShaders(ShaderReload& reload);
private:
    void release();
    std::uint32_t capacity_;
    bgfx::DynamicIndexBufferHandle counts_=BGFX_INVALID_HANDLE,free_=BGFX_INVALID_HANDLE,active_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,classify_=BGFX_INVALID_HANDLE,spawn_=BGFX_INVALID_HANDLE,brush_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle pool_=BGFX_INVALID_HANDLE,min_=BGFX_INVALID_HANDLE,max_=BGFX_INVALID_HANDLE,value_=BGFX_INVALID_HANDLE;
};
}
