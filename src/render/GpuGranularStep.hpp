#pragma once
#include "GpuSweptBounds.hpp"
#include "GpuVoxelMotion.hpp"
#include <memory>
#include "GpuCellOccupancy.hpp"
#include "GpuJacobiContacts.hpp"
#include "GpuVelocityReconstruction.hpp"
#include "GpuContactVelocity.hpp"
#include "core/GranularStep.hpp"
namespace pw {
// Owns all substeps and publishes only when the entire tick succeeds.
class GpuGranularStep {
public:
    GpuGranularStep(const std::filesystem::path& assets,std::uint32_t capacity,
        GranularStepProfile profile={},std::uint32_t iterations=4,float gravity=9.81f,bool latchFailures=false,AnalyticSphere obstacle={},bool voxelTerrain=false);
    ~GpuGranularStep();
    GpuGranularStep(const GpuGranularStep&)=delete;
    GpuGranularStep& operator=(const GpuGranularStep&)=delete;
    // Voxel mode requires a current context on every submit and uses the Basin
    // grid. Analytic mode rejects a context; no mode switch occurs mid-tick.
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,const VoxelContactContext* terrain=nullptr);
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    // Phase error mask, proposed speed caps, max contact degree, boundary events.
    // Voxel mode: the last counter includes prediction events and finish changes.
    bgfx::DynamicIndexBufferHandle status() const{return latchFailures_?latched_:status_;}
    // Current tick: contact-phase error mask, max candidates/query (includes self),
    // max hash probes/query, collected contact solves. Never a total pair count.
    bgfx::DynamicIndexBufferHandle neighborhoodStatus() const{return neighborhood_;}
    bgfx::DynamicIndexBufferHandle occupancyStatus() const{return occupancy_.status();}
    // Candidate low/high and directed-contact low/high; successful phases only.
    bgfx::DynamicIndexBufferHandle contactTotals() const{return totals_;}
    void resetFailures(std::uint16_t view);
    GranularStepPlan plan() const{return plan_;}
    // Diagnostic read-only snapshots for nonzero capacity after submit. Valid until
    // the next submit; consume in a later sequential view. No readback/allocation.
    // Force, swept prediction, corrected pose, reconstructed velocity, response, finish.
    std::array<bgfx::DynamicVertexBufferHandle,6> diagnosticStates() const {
        return {forced_,voxel_?prediction_:sweep_.output(),contacts_.output(),velocity_.output(),response_.output(),finished_};
    }
private:
    void release();
    void total(std::uint16_t view,bgfx::DynamicVertexBufferHandle records,bgfx::DynamicIndexBufferHandle status,float mode);
    void collect(std::uint16_t view,bgfx::DynamicIndexBufferHandle source,float phase,float mode);
    std::uint32_t capacity_,iterations_;
    GranularStepProfile profile_;
    GranularStepPlan plan_;
    float gravity_;
    bool latchFailures_;
    AnalyticSphere obstacle_;
    GpuCellOccupancy occupancy_;
    GpuSweptBounds sweep_;
    std::unique_ptr<GpuVoxelMotion> voxel_;
    bgfx::DynamicVertexBufferHandle prediction_=BGFX_INVALID_HANDLE;
    GpuJacobiContacts contacts_;
    GpuVelocityReconstruction velocity_;
    GpuContactVelocity response_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle forced_=BGFX_INVALID_HANDLE,finished_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle totals_=BGFX_INVALID_HANDLE,neighborhood_=BGFX_INVALID_HANDLE,zero_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE,phaseStatus_=BGFX_INVALID_HANDLE,latched_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle totalProgram_=BGFX_INVALID_HANDLE,clear_=BGFX_INVALID_HANDLE,force_=BGFX_INVALID_HANDLE,finish_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle collect_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE,latch_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle totalParams_=BGFX_INVALID_HANDLE,params_=BGFX_INVALID_HANDLE,step_=BGFX_INVALID_HANDLE,collection_=BGFX_INVALID_HANDLE,sphere_=BGFX_INVALID_HANDLE;
};
void runGpuGranularStepTest(const std::filesystem::path& assets);
}
