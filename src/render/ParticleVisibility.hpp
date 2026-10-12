#pragma once
#include "core/Frustum.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
#include <cstdint>
namespace pw {
class ParticleVisibility {
public:
    ParticleVisibility(const std::filesystem::path& assets,std::uint32_t capacity,bool allowIndirect=true);
    ~ParticleVisibility();
    ParticleVisibility(const ParticleVisibility&)=delete;
    ParticleVisibility& operator=(const ParticleVisibility&)=delete;
    void prepare(std::uint16_t view,bgfx::DynamicVertexBufferHandle instances,
                 bgfx::DynamicIndexBufferHandle active,bgfx::DynamicIndexBufferHandle pool,
                 const Frustum& planes,bool cull=true);
    bgfx::DynamicVertexBufferHandle instances() const {return instances_;}
    bgfx::DynamicIndexBufferHandle counts() const {return counts_;}
    bgfx::IndirectBufferHandle arguments() const {return indirect_;}
    bool supportsIndirect() const {return bgfx::isValid(indirect_);}
private:
    void release();
    std::uint32_t capacity_;
    bgfx::DynamicVertexBufferHandle instances_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle counts_=BGFX_INVALID_HANDLE;
    bgfx::IndirectBufferHandle indirect_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,cull_=BGFX_INVALID_HANDLE,args_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle config_=BGFX_INVALID_HANDLE,planes_=BGFX_INVALID_HANDLE;
};
}
