#pragma once
#include "GpuVoxelMotion.hpp"
#include "GpuGranularStep.hpp"
#include "GpuSettling.hpp"
#include "assets/Materials.hpp"
#include "PoolDiagnostics.hpp"
namespace pw {
// Fixed basin particles, optionally using the complete pair/terrain tick.
// No command pool, deposition or physical terrain conversion yet.
class VoxelProbes {
public:
    static constexpr std::uint32_t count=256;
    VoxelProbes(const std::filesystem::path& assets,const MaterialLibrary& materials,bool interacting=false);
    ~VoxelProbes();
    VoxelProbes(const VoxelProbes&)=delete;VoxelProbes& operator=(const VoxelProbes&)=delete;
    void tick(std::uint16_t view,const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source);
    void prepare(std::uint16_t view);
    void render(std::uint16_t view);
    void poll(std::uint32_t frame){diagnostics_->poll(frame);if(settlingDiagnostics_)settlingDiagnostics_->poll(frame);}
    void request(std::uint16_t view){diagnostics_->request(view,latched_,ticks_);if(settlingDiagnostics_)settlingDiagnostics_->request(view,settling_->status(),ticks_);}
    std::uint32_t errors() const{return diagnostics_->latest().active;}
    bool fault() const{return diagnostics_->latest().valid && diagnostics_->latest().active!=0;}
    std::uint32_t contactDegree() const{return granular_?diagnostics_->latest().reserved:0;}
    bool hasSettling() const{return bool(settling_);}
    const PoolSnapshot& settlingSnapshot() const{return settlingDiagnostics_->latest();}
    bgfx::DynamicVertexBufferHandle settlingHistory() const{return settling_?settling_->history():bgfx::DynamicVertexBufferHandle{bgfx::kInvalidHandle};}
    std::uint64_t ticks() const{return ticks_;}
    bgfx::DynamicVertexBufferHandle state() const{return states_[current_];}
    bgfx::DynamicIndexBufferHandle status() const{return latched_;}
private:
    void release();
    GpuVoxelMotion motion_;
    std::unique_ptr<GpuGranularStep> granular_;
    std::unique_ptr<PoolDiagnostics> diagnostics_,settlingDiagnostics_;
    std::unique_ptr<GpuSettling> settling_;
    std::uint32_t materials_,current_=0;
    std::uint64_t ticks_=0;
    std::array<bgfx::DynamicVertexBufferHandle,2> states_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle forced_=BGFX_INVALID_HANDLE,renderCopy_=BGFX_INVALID_HANDLE,instances_=BGFX_INVALID_HANDLE,materialBuffer_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle zero_=BGFX_INVALID_HANDLE,phase_=BGFX_INVALID_HANDLE,latched_=BGFX_INVALID_HANDLE;
    bgfx::VertexBufferHandle vertices_=BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle force_=BGFX_INVALID_HANDLE,clear_=BGFX_INVALID_HANDLE,latch_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE,pack_=BGFX_INVALID_HANDLE,draw_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,gravity_=BGFX_INVALID_HANDLE,sim_=BGFX_INVALID_HANDLE;
};
void runVoxelProbesTest(const std::filesystem::path& assets);
}
