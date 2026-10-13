#pragma once
#include "GpuBlastImpulse.hpp"
#include "VoxelProbes.hpp"
#include "VoxelPublication.hpp"
namespace pw {
class PendingBlastImpulse {
public:
    enum class State {Prepared,Waiting,Ready,Published,Rejected,Stale};
    PendingBlastImpulse(const std::filesystem::path& assets,VoxelProbes& particles,VoxelPublicationPtr terrain,
                        const MaterialLibrary& materials,BlastImpulseRequest request);
    ~PendingBlastImpulse();
    PendingBlastImpulse(const PendingBlastImpulse&)=delete;
    PendingBlastImpulse& operator=(const PendingBlastImpulse&)=delete;
    void submit(std::uint16_t view,VoxelProbes& particles);
    void poll(std::uint32_t frame);
    bool publish(std::uint16_t view,VoxelProbes& particles,const VoxelPublicationPtr& terrain);
    State state() const{return state_;}
    std::uint32_t errors() const{return errors_;}
    std::uint32_t accepted() const{return accepted_;}
    // A validated proposed charge in Ready; charge it only when publish succeeds.
    double energyJoules() const{return energy_;}
private:
    bool matches(const VoxelProbes& particles) const;
    void release();
    std::shared_ptr<const int> lease_;
    std::weak_ptr<const int> owner_;
    VoxelPublicationPtr terrain_;
    std::uint64_t tick_=0,materialHash_=0;
    std::uint32_t capacity_=0,errors_=0,accepted_=0;
    double energy_=0;
    State state_=State::Prepared;
    bgfx::DynamicVertexBufferHandle source_=BGFX_INVALID_HANDLE;
    std::unique_ptr<GpuBlastImpulse> impulse_;
    std::unique_ptr<PoolDiagnostics> acknowledgement_;
    bgfx::ProgramHandle acknowledge_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
};
void runPendingBlastImpulseTest(const std::filesystem::path& assets);
}
