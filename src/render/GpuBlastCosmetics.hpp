#pragma once
#include "core/BlastCosmetics.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
namespace pw {
// Owns visual-only storage. No physical pool, terrain or material handle enters this API.
class GpuBlastCosmetics {
public:
    GpuBlastCosmetics(const std::filesystem::path& assets,std::uint32_t dustCapacity=512,std::uint32_t sparkCapacity=256);
    ~GpuBlastCosmetics();
    GpuBlastCosmetics(const GpuBlastCosmetics&)=delete;
    GpuBlastCosmetics& operator=(const GpuBlastCosmetics&)=delete;
    void reset(std::uint16_t view);
    void emit(std::uint16_t view,const BlastCosmeticEvent& event,BlastCosmeticQuota quota={});
    void step(std::uint16_t view,float dt=1.f/60);
    void render(std::uint16_t view);
    std::uint32_t capacity()const{return dust_+sparks_;}
    bgfx::DynamicVertexBufferHandle state()const{return state_;}
    bgfx::DynamicVertexBufferHandle instances()const{return instances_;}
    // active dust, active sparks, cumulative dropped dust, cumulative dropped sparks.
    bgfx::DynamicIndexBufferHandle counts()const{return counts_;}
private:
    void release();void validateView(std::uint16_t view)const;
    std::uint32_t dust_,sparks_;
    bgfx::DynamicVertexBufferHandle state_=BGFX_INVALID_HANDLE,instances_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle counts_=BGFX_INVALID_HANDLE;
    bgfx::VertexBufferHandle vertices_=BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle reset_=BGFX_INVALID_HANDLE,spawn_=BGFX_INVALID_HANDLE,clear_=BGFX_INVALID_HANDLE,step_=BGFX_INVALID_HANDLE,draw_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuBlastCosmeticsTest(const std::filesystem::path& assets);
}
