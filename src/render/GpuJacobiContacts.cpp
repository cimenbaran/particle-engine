#include "GpuJacobiContacts.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace pw {
GpuJacobiContacts::GpuJacobiContacts(const std::filesystem::path& assets,std::uint32_t capacity,float relaxation,float maxCorrection,bool bounded,AnalyticSphere obstacle,GranularGridProfile profile)
    :capacity_(capacity),relaxation_(relaxation),maxCorrection_(maxCorrection),bounded_(bounded),obstacle_(obstacle),keys_(assets,capacity,profile),sort_(assets,capacity),ranges_(assets,capacity,profile),hash_(assets,capacity,0,64,profile) {
    validateAnalyticSphere(obstacle_);
    if(obstacle_.radius>0 && !bounded_)throw std::invalid_argument("Sphere contacts require bounded profile");
    if(!std::isfinite(relaxation) || relaxation<=0 || relaxation>1 || !std::isfinite(maxCorrection) || maxCorrection<=0 || maxCorrection>radius)
        throw std::invalid_argument("Invalid Jacobi relaxation/correction bound");
    try {
        bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:states_) h=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*4,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        corrections_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*2,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        packed_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*8,layout,BGFX_BUFFER_COMPUTE_READ_WRITE);
        status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        clear_=loadCompute(assets,"cs_contact_clear");solve_=loadCompute(assets,gridShader("cs_contacts",profile));
        reduce_=loadCompute(assets,"cs_contact_reduce");apply_=loadCompute(assets,"cs_contact_apply");
        pack_=loadCompute(assets,gridShader("cs_contact_pack",profile));
        terrain_=loadCompute(assets,gridShader("cs_voxel_contacts",profile));
        neighborParams_=bgfx::createUniform("u_neighbors",bgfx::UniformType::Vec4);
        contactParams_=bgfx::createUniform("u_contacts",bgfx::UniformType::Vec4);
        boundsParams_=bgfx::createUniform("u_contact_bounds",bgfx::UniformType::Vec4);
        sphereParams_=bgfx::createUniform("u_contact_sphere",bgfx::UniformType::Vec4);
        floorParams_=bgfx::createUniform("u_contact_floor",bgfx::UniformType::Vec4);
        frictionParams_=bgfx::createUniform("u_contact_friction",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(states_[0]) || !bgfx::isValid(states_[1]) || !bgfx::isValid(corrections_) ||
           !bgfx::isValid(sphereParams_) || !bgfx::isValid(floorParams_) || !bgfx::isValid(packed_) || !bgfx::isValid(frictionParams_) || !bgfx::isValid(status_) || !bgfx::isValid(neighborParams_) || !bgfx::isValid(contactParams_) || !bgfx::isValid(boundsParams_))
            throw std::runtime_error("Jacobi resources unavailable");
    }catch(...) {release();throw;}
}
GpuJacobiContacts::~GpuJacobiContacts(){release();}
void GpuJacobiContacts::release() {
    for(auto h:states_) if(bgfx::isValid(h)) bgfx::destroy(h);
    if(bgfx::isValid(corrections_)) bgfx::destroy(corrections_);
    if(bgfx::isValid(packed_)) bgfx::destroy(packed_);
    if(bgfx::isValid(status_)) bgfx::destroy(status_);
    for(auto h:{clear_,solve_,reduce_,apply_,pack_,terrain_}) if(bgfx::isValid(h)) bgfx::destroy(h);
    for(auto h:{neighborParams_,contactParams_,boundsParams_,frictionParams_,floorParams_,sphereParams_}) if(bgfx::isValid(h)) bgfx::destroy(h);
}
void GpuJacobiContacts::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,std::uint32_t materialCount,bgfx::DynamicVertexBufferHandle start,FloorTractionProfile floor,const VoxelContactContext* terrain) {
    if(terrain) {
        if(!terrain->world.isCurrent(terrain->source,terrain->origin) || obstacle_.radius>0)
            throw std::invalid_argument("Stale or incompatible Jacobi terrain context");
        const auto h=terrain->world.buffer();
        if(h.idx==input.idx || h.idx==materials.idx || h.idx==start.idx)
            throw std::invalid_argument("Jacobi terrain input alias");
    }
    if(!std::isfinite(floor.gravity) || floor.gravity<0 || floor.gravity>100 || !std::isfinite(floor.substepSeconds) || floor.substepSeconds<0 || floor.substepSeconds>1 ||
       !std::isfinite(floor.speedLimit) || floor.speedLimit<0.001f || floor.speedLimit>1000000 ||
       (floor.gravity>0 && ((!bounded_ && !terrain) || !bgfx::isValid(start) || floor.substepSeconds<std::numeric_limits<float>::min())))
        throw std::invalid_argument("Invalid floor traction profile");
    if(terrain && floor.gravity>0 && (floor.substepSeconds>.01f || floor.speedLimit>12))
        throw std::invalid_argument("Voxel traction requires substep <=.01s and speed <=12m/s");
    if(!bgfx::isValid(input) || !bgfx::isValid(materials) || input.idx==materials.idx || input.idx==corrections_.idx ||
       materials.idx==corrections_.idx || materials.idx==states_[0].idx || materials.idx==states_[1].idx || !materialCount || materialCount>65536)
        throw std::invalid_argument("Invalid Jacobi input or material buffer");
    if(input.idx==packed_.idx || materials.idx==packed_.idx)throw std::invalid_argument("Jacobi packed scratch alias");
    if(bgfx::isValid(start)) {
        if(start.idx==materials.idx || start.idx==packed_.idx || start.idx==corrections_.idx || start.idx==states_[0].idx || start.idx==states_[1].idx)
            throw std::invalid_argument("Jacobi substep baseline must be immutable external storage");
    }
    output_=input.idx==states_[0].idx?states_[1]:states_[0];
    keys_.submit(view,input,materials,materialCount);sort_.submit(view,keys_.records());
    ranges_.submit(view,sort_.output(),keys_.status());hash_.submit(view,ranges_.ranges(),ranges_.metadata());
    bgfx::setBuffer(0,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(1,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
    if(!capacity_) return;
    const float floorParameters[]{floor.gravity,floor.substepSeconds,floor.speedLimit,terrain?1.0f:0.0f};bgfx::setUniform(floorParams_,floorParameters);
    const float friction[]{float(capacity_),float(materialCount),bgfx::isValid(start)?1.0f:0.0f,0};
    bgfx::setUniform(frictionParams_,friction);bgfx::setBuffer(0,input,bgfx::Access::Read);
    bgfx::setBuffer(1,bgfx::isValid(start)?start:input,bgfx::Access::Read);bgfx::setBuffer(2,materials,bgfx::Access::Read);
    bgfx::setBuffer(3,packed_,bgfx::Access::Write);bgfx::dispatch(view,pack_,dispatchGroups(capacity_));
    const float neighbors[]{float(capacity_),float(hash_.tableSlots()),float(hash_.probeLimit()),radius*2};
    const float contacts[]{float(capacity_),radius,relaxation_,maxCorrection_};
    const float sphere[]{obstacle_.centre[0],obstacle_.centre[1],obstacle_.centre[2],obstacle_.radius};bgfx::setUniform(sphereParams_,sphere);
    const float bounds[]{bounded_?1.0f:0.0f,0,0,0};bgfx::setUniform(boundsParams_,bounds);
    bgfx::setUniform(neighborParams_,neighbors);bgfx::setUniform(contactParams_,contacts);
    bgfx::setBuffer(0,packed_,bgfx::Access::Read);bgfx::setBuffer(1,sort_.output(),bgfx::Access::Read);
    bgfx::setBuffer(2,ranges_.ranges(),bgfx::Access::Read);bgfx::setBuffer(3,hash_.table(),bgfx::Access::Read);
    bgfx::setBuffer(4,hash_.metadata(),bgfx::Access::Read);bgfx::setBuffer(5,corrections_,bgfx::Access::ReadWrite);
    bgfx::dispatch(view,solve_,dispatchGroups(capacity_));
    if(terrain) {
        bgfx::setUniform(contactParams_,contacts);bgfx::setUniform(boundsParams_,bounds);bgfx::setUniform(frictionParams_,friction);
        bgfx::setBuffer(0,packed_,bgfx::Access::Read);bgfx::setBuffer(1,corrections_,bgfx::Access::ReadWrite);
        bgfx::setBuffer(2,terrain->world.buffer(),bgfx::Access::Read);bgfx::setBuffer(3,materials,bgfx::Access::Read);
        bgfx::dispatch(view,terrain_,dispatchGroups(capacity_));
    }
    bgfx::setUniform(contactParams_,contacts);bgfx::setBuffer(0,corrections_,bgfx::Access::Read);
    bgfx::setBuffer(1,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,reduce_,dispatchGroups(capacity_));
    bgfx::setUniform(contactParams_,contacts);bgfx::setBuffer(0,input,bgfx::Access::Read);
    bgfx::setBuffer(1,corrections_,bgfx::Access::Read);bgfx::setBuffer(2,status_,bgfx::Access::Read);
    bgfx::setBuffer(3,output_,bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(capacity_));
}
}
