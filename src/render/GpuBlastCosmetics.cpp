#include "GpuBlastCosmetics.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
namespace pw {
GpuBlastCosmetics::GpuBlastCosmetics(const std::filesystem::path& assets,std::uint32_t dust,std::uint32_t sparks):dust_(dust),sparks_(sparks){
    if(dust>2048 || sparks>2048 || dust+sparks==0)throw std::invalid_argument("Cosmetic capacities require 1..4096 total slots, at most 2048 per family");
    try{
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        state_=bgfx::createDynamicVertexBuffer(capacity()*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        bgfx::VertexLayout instances;instances.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).add(bgfx::Attrib::TexCoord1,4,bgfx::AttribType::Float).end();
        instances_=bgfx::createDynamicVertexBuffer(capacity(),instances,BGFX_BUFFER_COMPUTE_READ_WRITE);
        counts_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        bgfx::VertexLayout positions;positions.begin().add(bgfx::Attrib::Position,3,bgfx::AttribType::Float).end();
        const float quad[]{-1,-1,0,1,-1,0,1,1,0,-1,1,0};const std::uint16_t indices[]{0,1,2,0,2,3};
        vertices_=bgfx::createVertexBuffer(bgfx::copy(quad,sizeof(quad)),positions);indices_=bgfx::createIndexBuffer(bgfx::copy(indices,sizeof(indices)));
        reset_=loadCompute(assets,"cs_cosmetic_reset");spawn_=loadCompute(assets,"cs_cosmetic_spawn");clear_=loadCompute(assets,"cs_cosmetic_counts");step_=loadCompute(assets,"cs_cosmetic_step");draw_=loadProgram(assets,"vs_particles","fs_cosmetics");
        params_=bgfx::createUniform("u_cosmetic",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(state_) || !bgfx::isValid(instances_) || !bgfx::isValid(counts_) || !bgfx::isValid(vertices_) || !bgfx::isValid(indices_) || !bgfx::isValid(params_))throw std::runtime_error("Cosmetic GPU allocation failed");
        reset(0);
    }catch(...){release();throw;}
}
void GpuBlastCosmetics::release(){
    if(bgfx::isValid(state_))bgfx::destroy(state_);if(bgfx::isValid(instances_))bgfx::destroy(instances_);if(bgfx::isValid(counts_))bgfx::destroy(counts_);
    if(bgfx::isValid(vertices_))bgfx::destroy(vertices_);if(bgfx::isValid(indices_))bgfx::destroy(indices_);
    for(auto h:{reset_,spawn_,clear_,step_,draw_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
GpuBlastCosmetics::~GpuBlastCosmetics(){release();}
void GpuBlastCosmetics::validateView(std::uint16_t view)const{if(view>=bgfx::getCaps()->limits.maxViews)throw std::invalid_argument("Invalid cosmetic view");}
void GpuBlastCosmetics::reset(std::uint16_t view){
    validateView(view);bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity()),float(dust_),0,0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,state_,bgfx::Access::Write);bgfx::setBuffer(1,instances_,bgfx::Access::Write);bgfx::setBuffer(2,counts_,bgfx::Access::Write);bgfx::dispatch(view,reset_,dispatchGroups(capacity()));
}
void GpuBlastCosmetics::emit(std::uint16_t view,const BlastCosmeticEvent& event,BlastCosmeticQuota quota){
    validateView(view);const auto seeds=planBlastCosmetics(event,quota);if(seeds.empty())return;
    std::vector<ParticleGpu> input(seeds.size());
    for(unsigned i=0;i<input.size();++i){const auto& s=seeds[i];auto& p=input[i];for(unsigned a=0;a<3;++a){p.positionRadius[a]=s.position[a];p.velocityInvMass[a]=s.velocity[a];}p.positionRadius[3]=s.radius;p.thermalState[2]=s.lifetime;p.identity[0]=std::uint32_t(s.kind);}
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    const auto upload=bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),unsigned(input.size()*sizeof(ParticleGpu))),lanes,BGFX_BUFFER_COMPUTE_READ);
    if(!bgfx::isValid(upload))throw std::runtime_error("Cosmetic seed upload failed");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);const float p[]{float(capacity()),float(dust_),0,float(input.size())};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,state_,bgfx::Access::ReadWrite);bgfx::setBuffer(1,upload,bgfx::Access::Read);bgfx::setBuffer(2,counts_,bgfx::Access::ReadWrite);bgfx::dispatch(view,spawn_,1);
    bgfx::destroy(upload);step(view,0);
}
void GpuBlastCosmetics::step(std::uint16_t view,float dt){
    validateView(view);if(!std::isfinite(dt) || dt<0 || dt>1.f/30)throw std::invalid_argument("Invalid cosmetic step");
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setBuffer(0,counts_,bgfx::Access::ReadWrite);bgfx::dispatch(view,clear_,1);
    const float p[]{float(capacity()),float(dust_),dt,0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,state_,bgfx::Access::ReadWrite);bgfx::setBuffer(1,instances_,bgfx::Access::Write);bgfx::setBuffer(2,counts_,bgfx::Access::ReadWrite);bgfx::dispatch(view,step_,dispatchGroups(capacity()));
}
void GpuBlastCosmetics::render(std::uint16_t view){validateView(view);bgfx::setVertexBuffer(0,vertices_);bgfx::setIndexBuffer(indices_);bgfx::setInstanceDataBuffer(instances_,0,capacity());bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_BLEND_ALPHA|BGFX_STATE_MSAA);bgfx::submit(view,draw_);}
}
