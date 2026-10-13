#include "GpuContactNormals.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <stdexcept>
namespace pw {
GpuContactNormals::GpuContactNormals(const std::filesystem::path& assets,std::uint32_t capacity)
    :capacity_(capacity),keys_(assets,capacity),sort_(assets,capacity),ranges_(assets,capacity),hash_(assets,capacity) {
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        output_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        program_=loadCompute(assets,"cs_contact_normals");
        neighbors_=bgfx::createUniform("u_neighbors",bgfx::UniformType::Vec4);
        sphere_=bgfx::createUniform("u_normal_sphere",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(output_) || !bgfx::isValid(neighbors_) || !bgfx::isValid(sphere_))throw std::runtime_error("Contact normal resources unavailable");
    }catch(...){release();throw;}
}
GpuContactNormals::~GpuContactNormals(){release();}
void GpuContactNormals::release(){
    if(bgfx::isValid(output_))bgfx::destroy(output_);
    if(bgfx::isValid(program_))bgfx::destroy(program_);
    for(auto h:{neighbors_,sphere_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void GpuContactNormals::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle state,
    bgfx::DynamicVertexBufferHandle materials,std::uint32_t count,AnalyticSphere sphere) {
    validateAnalyticSphere(sphere);
    if(!bgfx::isValid(state) || !bgfx::isValid(materials) || state.idx==materials.idx ||
       state.idx==output_.idx || materials.idx==output_.idx || !count || count>65536)
        throw std::invalid_argument("Invalid contact normal inputs");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    keys_.submit(view,state,materials,count);sort_.submit(view,keys_.records());
    ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    if(!capacity_)return;
    const float n[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),0.127f};
    const float s[]{sphere.centre[0],sphere.centre[1],sphere.centre[2],sphere.radius};
    bgfx::setUniform(neighbors_,n);bgfx::setUniform(sphere_,s);
    bgfx::setBuffer(0,state,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);
    bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);
    bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,output_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,program_,dispatchGroups(capacity_));
}
}
