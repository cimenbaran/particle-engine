#pragma once
#include "GpuPrefixScan.hpp"
#include <array>
namespace pw {
struct BlastImpulseRequest {
    std::array<float,3> centre{};
    float radius=1,speed=2;
    double joules=0;
};
// Private candidate only. The caller must freeze/validate the source and obtain
// an acknowledgement before adopting output or charging its energy ticket total.
class GpuBlastImpulse {
public:
    GpuBlastImpulse(const std::filesystem::path& assets,std::uint32_t capacity,BlastImpulseRequest request);
    ~GpuBlastImpulse();
    GpuBlastImpulse(const GpuBlastImpulse&)=delete;
    GpuBlastImpulse& operator=(const GpuBlastImpulse&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle source,bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount);
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    // errors (1 identity, 2 numeric, 4 material), eligible, accepted, tickets used.
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
    double joulesPerTicket() const{return quantum_;}
    std::uint32_t ticketBudget() const{return tickets_;}
    static constexpr std::uint32_t maximumTickets=0x00ffffffu;
private:
    friend class PendingBlastImpulse;
    void release();
    std::uint32_t capacity_,tickets_=0;
    BlastImpulseRequest request_;
    float quantum_=1;
    bool submitted_=false;
    GpuPrefixScan scan_;
    bgfx::DynamicVertexBufferHandle candidate_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle costs_=BGFX_INVALID_HANDLE,status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,propose_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,centre_=BGFX_INVALID_HANDLE,speed_=BGFX_INVALID_HANDLE;
};
void runGpuBlastImpulseTest(const std::filesystem::path& assets);
}
