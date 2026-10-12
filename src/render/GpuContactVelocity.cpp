#include "GpuContactVelocity.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace pw {
namespace {
std::uint32_t validate(std::uint32_t capacity,float bound,float slop,float threshold) {
    if(capacity>1048576 || !std::isfinite(bound) || bound<0 || bound>32 || !std::isfinite(slop) || slop<0 || slop>0.0625f ||
       !std::isfinite(threshold) || threshold<0) throw std::invalid_argument("Invalid contact velocity profile");
    return capacity;
}
}
GpuContactVelocity::GpuContactVelocity(const std::filesystem::path& assets,std::uint32_t capacity,float bound,float slop,float threshold)
    :capacity_(validate(capacity,bound,slop,threshold)),bound_(bound+1e-5f),slop_(slop),threshold_(threshold),
     keys_(assets,capacity_),sort_(assets,capacity_),ranges_(assets,capacity_),hash_(assets,capacity_) {
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:outputs_) h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        packed_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*8,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        result_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*2,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_velocity_clear");pack_=loadCompute(assets,"cs_response_pack");degree_=loadCompute(assets,"cs_response_degree");
        respond_=loadCompute(assets,"cs_response_velocity");reduce_=loadCompute(assets,"cs_response_reduce");apply_=loadCompute(assets,"cs_response_apply");
        params_=bgfx::createUniform("u_response",bgfx::UniformType::Vec4);neighbors_=bgfx::createUniform("u_neighbors",bgfx::UniformType::Vec4);
        materials_=bgfx::createUniform("u_response_materials",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(outputs_[0]) || !bgfx::isValid(outputs_[1]) || !bgfx::isValid(packed_) || !bgfx::isValid(result_) ||
           !bgfx::isValid(status_) || !bgfx::isValid(params_) || !bgfx::isValid(neighbors_) || !bgfx::isValid(materials_))
            throw std::runtime_error("Contact velocity resources unavailable");
    }catch(...) {release();throw;}
}
GpuContactVelocity::~GpuContactVelocity(){release();}
void GpuContactVelocity::release() {
    for(auto h:outputs_) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{packed_,result_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(status_)) bgfx::destroy(status_);
    for(auto h:{clear_,pack_,degree_,respond_,reduce_,apply_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{params_,neighbors_,materials_}) if(bgfx::isValid(h)) bgfx::destroy(h);
}
void GpuContactVelocity::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle predicted,bgfx::DynamicVertexBufferHandle current,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream) {
    if(!bgfx::isValid(predicted) || !bgfx::isValid(current) || !bgfx::isValid(materials) || !bgfx::isValid(upstream) ||
       !materialCount || materialCount>65536 || upstream.idx==status_.idx || materials.idx==predicted.idx || materials.idx==current.idx)
        throw std::invalid_argument("Invalid contact velocity input");
    for(auto h:{packed_,result_}) if(h.idx==predicted.idx || h.idx==current.idx || h.idx==materials.idx)
        throw std::invalid_argument("Contact velocity input aliases scratch");
    for(auto h:outputs_) if(h.idx==materials.idx) throw std::invalid_argument("Contact velocity material/output alias");
    auto output=outputs_[0];if(output.idx==predicted.idx || output.idx==current.idx) output=outputs_[1];
    if(output.idx==predicted.idx || output.idx==current.idx) throw std::invalid_argument("No separate contact velocity output available");
    output_=output;bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),bound_,slop_,threshold_},m[]{float(materialCount),0,0,0};
    const float n[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),std::max(0.125f+2*bound_,0.125f+slop_)};
    bgfx::setBuffer(0,upstream,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(capacity_) {
        bgfx::setUniform(params_,p);bgfx::setUniform(materials_,m);
        bgfx::setBuffer(0,predicted,bgfx::Access::Read);bgfx::setBuffer(1,current,bgfx::Access::Read);bgfx::setBuffer(2,materials,bgfx::Access::Read);
        bgfx::setBuffer(3,upstream,bgfx::Access::Read);bgfx::setBuffer(4,packed_,bgfx::Access::Write);bgfx::setBuffer(5,status_,bgfx::Access::ReadWrite);
        bgfx::dispatch(view,pack_,dispatchGroups(capacity_));
    }
    keys_.submit(view,current,materials,materialCount);sort_.submit(view,keys_.records());
    ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    if(!capacity_) return;
    bgfx::setUniform(params_,p);bgfx::setUniform(neighbors_,n);
    bgfx::setBuffer(0,packed_,bgfx::Access::ReadWrite);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);
    bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);
    bgfx::dispatch(view,degree_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setUniform(neighbors_,n);
    bgfx::setBuffer(0,packed_,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);
    bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);
    bgfx::setBuffer(5,result_,bgfx::Access::ReadWrite);bgfx::dispatch(view,respond_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,result_,bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,reduce_,dispatchGroups(capacity_));
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,current,bgfx::Access::Read);bgfx::setBuffer(1,result_,bgfx::Access::Read);
    bgfx::setBuffer(2,packed_,bgfx::Access::Read);bgfx::setBuffer(3,status_,bgfx::Access::Read);bgfx::setBuffer(4,output_,bgfx::Access::Write);
    bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
