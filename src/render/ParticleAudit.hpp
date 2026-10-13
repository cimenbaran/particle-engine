#pragma once
#include "PoolDiagnostics.hpp"
#include "core/AnalyticSphere.hpp"
#include <memory>
namespace pw {
class ParticleAudit {
public:
    ParticleAudit(const std::filesystem::path& assets,std::uint32_t capacity);
    ~ParticleAudit();
    ParticleAudit(const ParticleAudit&)=delete;
    ParticleAudit& operator=(const ParticleAudit&)=delete;
    void reset();
    // Optional sphere audits every live body against its expanded surface.
    void prepare(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
                 bgfx::DynamicIndexBufferHandle active,bgfx::DynamicIndexBufferHandle free,
                 bgfx::DynamicIndexBufferHandle pool,std::uint32_t materials,AnalyticSphere sphere={});
    bool request(std::uint16_t view,std::uint64_t tick) {return readback_->request(view,result_,tick);}
    void poll(std::uint32_t frame) {readback_->poll(frame);}
    unsigned pending() const {return readback_->pending();}
    // active=checked slots; free/reserved/rejected=latched ownership/state/identity flags.
    const PoolSnapshot& latest() const {return readback_->latest();}
private:
    void release();
    std::uint32_t capacity_;
    bool clearErrors_=true;
    std::unique_ptr<PoolDiagnostics> readback_;
    bgfx::DynamicIndexBufferHandle owners_=BGFX_INVALID_HANDLE,result_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,membership_=BGFX_INVALID_HANDLE,state_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle config_=BGFX_INVALID_HANDLE,sphere_=BGFX_INVALID_HANDLE;
};
void runGpuAuditTest(const std::filesystem::path& assets);
}
