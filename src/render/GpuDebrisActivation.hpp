#pragma once
#include "GpuDebrisReservation.hpp"
#include "assets/Materials.hpp"
#include <span>
namespace pw {
struct DebrisSeed {
    std::array<float,3> position{},velocity{};
    std::uint32_t material=0;
    float temperature=293.15f,fuel=0;
};
// One-shot initialization of a private reservation snapshot. This is not a
// commit or a geometric clearance test. Keep the output private until terrain,
// particle clearance, base-tick freshness and mass gates have all succeeded.
class GpuDebrisActivation {
public:
    GpuDebrisActivation(const std::filesystem::path& assets,std::uint32_t capacity,
                       const MaterialLibrary& materials,std::span<const DebrisSeed> seeds);
    ~GpuDebrisActivation();
    GpuDebrisActivation(const GpuDebrisActivation&)=delete;
    GpuDebrisActivation& operator=(const GpuDebrisActivation&)=delete;
    // Reservation buffers must remain unchanged/alive through activation and any
    // subsequent clearance execution (claims() is a borrowed handle).
    // Rejects a second submission; immutable seed upload cannot be overwritten.
    void submit(std::uint16_t view,const GpuDebrisReservation& reservation);
    bool submitted() const{return submitted_;}
    std::uint32_t capacity() const{return capacity_;}
    std::uint32_t count() const{return count_;}
    bgfx::DynamicVertexBufferHandle claims() const{return sourceClaims_;}
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    // errors: 1 reservation/count, 2 claim/generation/ownership. Then seed count,
    // reserved slots observed, activated count (zero for any error).
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
    std::uint64_t materialHash() const{return materialHash_;}
private:
    friend class PendingDebrisTransaction;
    void release();
    std::uint32_t capacity_,count_;
    std::uint64_t materialHash_;
    bool submitted_=false;
    bgfx::DynamicVertexBufferHandle sourceClaims_=BGFX_INVALID_HANDLE,seeds_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle owners_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,ownersClear_=BGFX_INVALID_HANDLE,claims_=BGFX_INVALID_HANDLE,finalize_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuDebrisActivationTest(const std::filesystem::path& assets);
}
