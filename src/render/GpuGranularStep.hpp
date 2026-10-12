#pragma once
#include "GpuSweptBounds.hpp"
#include "GpuJacobiContacts.hpp"
#include "GpuVelocityReconstruction.hpp"
#include "GpuContactVelocity.hpp"
#include "core/GranularStep.hpp"
namespace pw {
// Owns all substeps and publishes only when the entire tick succeeds.
class GpuGranularStep {
public:
    GpuGranularStep(const std::filesystem::path& assets,std::uint32_t capacity,
        GranularStepProfile profile={},std::uint32_t iterations=4,float gravity=9.81f,bool latchFailures=false);
    ~GpuGranularStep();
    GpuGranularStep(const GpuGranularStep&)=delete;
    GpuGranularStep& operator=(const GpuGranularStep&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,
        bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount);
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    // Phase error mask, proposed speed caps, max contact degree, boundary events.
    bgfx::DynamicIndexBufferHandle status() const{return latchFailures_?latched_:status_;}
    // Current tick: contact-phase error mask, max candidates/query (includes self),
    // max hash probes/query, collected contact solves. Never a total pair count.
    bgfx::DynamicIndexBufferHandle neighborhoodStatus() const{return neighborhood_;}
    void resetFailures(std::uint16_t view);
    GranularStepPlan plan() const{return plan_;}
private:
    void release();
    void collect(std::uint16_t view,bgfx::DynamicIndexBufferHandle source,float phase,float mode);
    std::uint32_t capacity_,iterations_;
    GranularStepProfile profile_;
    GranularStepPlan plan_;
    float gravity_;
    bool latchFailures_;
    GpuSweptBounds sweep_;
    GpuJacobiContacts contacts_;
    GpuVelocityReconstruction velocity_;
    GpuContactVelocity response_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle forced_=BGFX_INVALID_HANDLE,finished_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle neighborhood_=BGFX_INVALID_HANDLE,zero_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE,phaseStatus_=BGFX_INVALID_HANDLE,latched_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,force_=BGFX_INVALID_HANDLE,finish_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle collect_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE,latch_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,step_=BGFX_INVALID_HANDLE,collection_=BGFX_INVALID_HANDLE;
};
void runGpuGranularStepTest(const std::filesystem::path& assets);
}
