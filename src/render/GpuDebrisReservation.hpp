#pragma once
#include "GpuPrefixScan.hpp"
namespace pw {
// All-or-nothing allocation in a private transaction snapshot. Never mutates
// live input or terrain. Output is NOT a live pool: reserved identities carry
// flag 2 and must be activated/validated before coordinated publication.
class GpuDebrisReservation {
public:
    GpuDebrisReservation(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuDebrisReservation();
    GpuDebrisReservation(const GpuDebrisReservation&)=delete;
    GpuDebrisReservation& operator=(const GpuDebrisReservation&)=delete;
    // Input must stay immutable/alive through GPU execution. The transaction
    // coordinator must separately freeze or validate its base tick at commit.
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,std::uint32_t requested);
    // Outputs remain valid until the next submit; rows below capacity only.
    bool submitted() const{return submitted_;}
    std::uint32_t capacity() const{return capacity_;}
    bgfx::DynamicVertexBufferHandle snapshot() const{return snapshot_;}
    // One uvec4 per requested slot: slot, new generation, old generation, 0.
    // Sorted by slot. Unused rows within capacity are UINT_MAX,0,0,0.
    bgfx::DynamicVertexBufferHandle claims() const{return claims_;}
    // errors (1 insufficient capacity, 2 invalid identity), requested,
    // eligible slots, reserved count. No reservations on any error.
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
private:
    void release();
    std::uint32_t capacity_;
    bool submitted_=false;
    GpuPrefixScan scan_;
    bgfx::DynamicVertexBufferHandle snapshot_=BGFX_INVALID_HANDLE,claims_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle flags_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,flag_=BGFX_INVALID_HANDLE,finalize_=BGFX_INVALID_HANDLE,reserve_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE;
};
void runGpuDebrisReservationTest(const std::filesystem::path& assets);
}
