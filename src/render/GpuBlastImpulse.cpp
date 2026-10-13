#include "GpuBlastImpulse.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace pw {
namespace {
std::uint32_t validate(std::uint32_t capacity,const BlastImpulseRequest& request) {
    if(capacity>1048576 || !std::isfinite(request.radius) || request.radius<.0625f || request.radius>4 ||
       !std::isfinite(request.speed) || request.speed<0 || request.speed>12 || !std::isfinite(request.joules) ||
       request.joules<0 || request.joules>1e9 || (request.joules>0 && request.joules<1e-6))
        throw std::invalid_argument("Invalid blast impulse capacity, radius, speed or energy");
    for(unsigned a=0;a<3;++a)if(!std::isfinite(request.centre[a]) || request.centre[a]<(a==1?0:-16) || request.centre[a]>16)
        throw std::invalid_argument("Blast impulse centre outside basin");
    return capacity;
}
}
GpuBlastImpulse::GpuBlastImpulse(const std::filesystem::path& assets,std::uint32_t capacity,BlastImpulseRequest request)
    :capacity_(validate(capacity,request)),request_(request),scan_(assets,capacity_,GpuPrefixScan::Arithmetic::Saturating) {
    if(request.joules>0){tickets_=maximumTickets;quantum_=float(request.joules/maximumTickets);if(double(quantum_)*maximumTickets>request.joules)quantum_=std::nextafter(quantum_,0.f);}
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        candidate_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        output_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity_)*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        costs_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity_),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_blast_clear");propose_=loadCompute(assets,"cs_blast_propose");apply_=loadCompute(assets,"cs_blast_apply");
        params_=bgfx::createUniform("u_blast",bgfx::UniformType::Vec4);centre_=bgfx::createUniform("u_blast_centre",bgfx::UniformType::Vec4);speed_=bgfx::createUniform("u_blast_speed",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(candidate_) || !bgfx::isValid(output_) || !bgfx::isValid(costs_) || !bgfx::isValid(status_) || !bgfx::isValid(params_) || !bgfx::isValid(centre_) || !bgfx::isValid(speed_))throw std::runtime_error("Blast impulse allocation failed");
    }catch(...){release();throw;}
}
GpuBlastImpulse::~GpuBlastImpulse(){release();}
void GpuBlastImpulse::release() {
    for(auto h:{candidate_,output_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{costs_,status_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{clear_,propose_,apply_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,centre_,speed_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuBlastImpulse::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle source,bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount) {
    if(submitted_)throw std::logic_error("Blast impulse is single-use");
    if(view>=bgfx::getCaps()->limits.maxViews || !bgfx::isValid(source) || !bgfx::isValid(materials) || source.idx==materials.idx || !materialCount || materialCount>65536)
        throw std::invalid_argument("Invalid blast impulse inputs/view");
    for(auto h:{candidate_,output_})if(source.idx==h.idx || materials.idx==h.idx)throw std::invalid_argument("Blast impulse scratch alias");
    submitted_=true;bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    bgfx::setBuffer(0,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_)return;
    const float p[]{float(capacity_),float(materialCount),float(tickets_),quantum_},c[]{request_.centre[0],request_.centre[1],request_.centre[2],request_.radius},s[]{request_.speed,0,0,0};
    bgfx::setUniform(params_,p);bgfx::setUniform(centre_,c);bgfx::setUniform(speed_,s);
    bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,materials,bgfx::Access::Read);bgfx::setBuffer(2,candidate_,bgfx::Access::Write);bgfx::setBuffer(3,costs_,bgfx::Access::Write);bgfx::setBuffer(4,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,propose_,dispatchGroups(capacity_));
    scan_.submit(view,costs_);
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,source,bgfx::Access::Read);bgfx::setBuffer(1,candidate_,bgfx::Access::Read);bgfx::setBuffer(2,costs_,bgfx::Access::Read);bgfx::setBuffer(3,scan_.output(),bgfx::Access::Read);bgfx::setBuffer(4,output_,bgfx::Access::Write);bgfx::setBuffer(5,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
