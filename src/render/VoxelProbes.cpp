#include "VoxelProbes.hpp"
#include "Shaders.hpp"
#include "core/ParticleTypes.hpp"
#include "core/ComputeDispatch.hpp"
namespace pw {
VoxelProbes::VoxelProbes(const std::filesystem::path& assets,const MaterialLibrary& materials,bool interacting)
    :motion_(assets,count),materials_(std::uint32_t(materials.entries.size())) {
    std::uint32_t sand=UINT32_MAX;for(unsigned i=0;i<materials_;++i)if(materials.entries[i].name=="Sand")sand=i;
    if(sand==UINT32_MAX)throw std::invalid_argument("Terrain probes require Sand material");
    try {
        if(interacting)granular_=std::make_unique<GpuGranularStep>(assets,count,GranularStepProfile{},4,9.81f,true,AnalyticSphere{},true);
        diagnostics_=std::make_unique<PoolDiagnostics>(assets);
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        for(auto& h:states_)h=bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        forced_=bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        renderCopy_=bgfx::createDynamicVertexBuffer(count*4,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        bgfx::VertexLayout instance;instance.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).add(bgfx::Attrib::TexCoord1,4,bgfx::AttribType::Float).end();
        instances_=bgfx::createDynamicVertexBuffer(count,instance,BGFX_BUFFER_COMPUTE_READ_WRITE);
        const auto packed=materials.packed();materialBuffer_=bgfx::createDynamicVertexBuffer(bgfx::copy(packed.data(),std::uint32_t(packed.size()*sizeof(MaterialGpu))),lanes,BGFX_BUFFER_COMPUTE_READ);
        const std::array<std::uint32_t,4> zero{};zero_=bgfx::createDynamicIndexBuffer(bgfx::copy(zero.data(),16),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ);
        phase_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);latched_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
        force_=loadCompute(assets,"cs_granular_force");clear_=loadCompute(assets,"cs_velocity_clear");latch_=loadCompute(assets,"cs_granular_latch");apply_=loadCompute(assets,"cs_velocity_apply");pack_=loadCompute(assets,"cs_reset");draw_=loadProgram(assets,"vs_particles","fs_particles");
        params_=bgfx::createUniform("u_velocity",bgfx::UniformType::Vec4);gravity_=bgfx::createUniform("u_granular",bgfx::UniformType::Vec4);sim_=bgfx::createUniform("u_sim",bgfx::UniformType::Vec4);
        bgfx::VertexLayout positions;positions.begin().add(bgfx::Attrib::Position,3,bgfx::AttribType::Float).end();
        const float quad[]{-1,-1,0,1,-1,0,1,1,0,-1,1,0};const std::uint16_t indices[]{0,1,2,0,2,3};
        vertices_=bgfx::createVertexBuffer(bgfx::copy(quad,sizeof(quad)),positions);indices_=bgfx::createIndexBuffer(bgfx::copy(indices,sizeof(indices)));
        for(auto h:{states_[0],states_[1],forced_,renderCopy_,instances_,materialBuffer_})if(!bgfx::isValid(h))throw std::runtime_error("Voxel probe buffer allocation failed");
        if(!bgfx::isValid(zero_) || !bgfx::isValid(phase_) || !bgfx::isValid(latched_) || !bgfx::isValid(vertices_) || !bgfx::isValid(indices_) || !bgfx::isValid(params_) || !bgfx::isValid(gravity_) || !bgfx::isValid(sim_))throw std::runtime_error("Voxel probe resources unavailable");
        std::array<ParticleGpu,count> input{};
        for(unsigned i=0;i<count;++i) {
            auto& p=input[i];p.positionRadius[0]=-11.5f+float(i%16)*.5f;p.positionRadius[1]=6;p.positionRadius[2]=-7.5f+float(i/16)*.25f;p.positionRadius[3]=.0625f;
            if(interacting){p.positionRadius[0]=-6.5f+float(i%8)*.14f;p.positionRadius[1]=6+float(i/64)*.14f;p.positionRadius[2]=-5+float((i/8)%8)*.14f;}
            p.velocityInvMass[0]=2;p.velocityInvMass[3]=1.0f/(4.0f/3.0f*3.14159265f*.0625f*.0625f*.0625f*materials.entries[sand].gpu.mechanical[0]);p.thermalState[0]=293.15f;
            p.identity[0]=sand;p.identity[1]=1;p.identity[2]=i;p.identity[3]=1;
        }
        auto upload=bgfx::createDynamicVertexBuffer(bgfx::copy(input.data(),sizeof(input)),lanes,BGFX_BUFFER_COMPUTE_READ);
        if(!bgfx::isValid(upload))throw std::runtime_error("Voxel probe upload failed");
        bgfx::setViewMode(0,bgfx::ViewMode::Sequential);bgfx::setBuffer(0,zero_,bgfx::Access::Read);bgfx::setBuffer(1,latched_,bgfx::Access::Write);bgfx::dispatch(0,clear_,1);
        const float p[]{0,float(count),float(materials_),0};bgfx::setUniform(sim_,p);
        bgfx::setBuffer(0,upload,bgfx::Access::Read);bgfx::setBuffer(1,states_[0],bgfx::Access::Write);bgfx::setBuffer(2,instances_,bgfx::Access::Write);bgfx::setBuffer(3,materialBuffer_,bgfx::Access::Read);bgfx::dispatch(0,pack_,dispatchGroups(count));bgfx::destroy(upload);
    }catch(...){release();throw;}
}
VoxelProbes::~VoxelProbes(){diagnostics_.reset();release();}
void VoxelProbes::release(){
    for(auto h:{states_[0],states_[1],forced_,renderCopy_,instances_,materialBuffer_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{zero_,phase_,latched_})if(bgfx::isValid(h))bgfx::destroy(h);
    if(bgfx::isValid(vertices_))bgfx::destroy(vertices_);if(bgfx::isValid(indices_))bgfx::destroy(indices_);
    for(auto h:{force_,clear_,latch_,apply_,pack_,draw_})if(bgfx::isValid(h))bgfx::destroy(h);
    for(auto h:{params_,gravity_,sim_})if(bgfx::isValid(h))bgfx::destroy(h);
}
void VoxelProbes::tick(std::uint16_t view,const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source) {
    if(!world.isCurrent(source,{}))throw std::invalid_argument("Stale voxel probe world");
    if(granular_) {
        const VoxelContactContext terrain{world,source,{}};
        granular_->submit(view,states_[current_],materialBuffer_,materials_,&terrain);
        bgfx::setBuffer(0,granular_->status(),bgfx::Access::Read);bgfx::setBuffer(1,latched_,bgfx::Access::ReadWrite);bgfx::dispatch(view,latch_,1);
        const float p[]{float(count),1.0f/240,float(materials_),0};bgfx::setUniform(params_,p);
        bgfx::setBuffer(0,states_[current_],bgfx::Access::Read);bgfx::setBuffer(1,granular_->output(),bgfx::Access::Read);
        bgfx::setBuffer(2,latched_,bgfx::Access::Read);bgfx::setBuffer(3,states_[1-current_],bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(count));
        current_=1-current_;++ticks_;return;
    }
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);auto current=states_[current_];
    const float p[]{float(count),1.0f/240,float(materials_),0},q[]{9.81f,12,0,0};
    for(unsigned step=0;step<4;++step) {
        bgfx::setBuffer(0,latched_,bgfx::Access::Read);bgfx::setBuffer(1,phase_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
        bgfx::setUniform(params_,p);bgfx::setUniform(gravity_,q);bgfx::setBuffer(0,current,bgfx::Access::Read);bgfx::setBuffer(1,materialBuffer_,bgfx::Access::Read);bgfx::setBuffer(2,latched_,bgfx::Access::Read);bgfx::setBuffer(3,forced_,bgfx::Access::Write);bgfx::setBuffer(4,phase_,bgfx::Access::ReadWrite);bgfx::dispatch(view,force_,dispatchGroups(count));
        motion_.submit(view,forced_,materialBuffer_,materials_,phase_,world,source,{},1.0f/240);current=motion_.output();
        bgfx::setBuffer(0,motion_.status(),bgfx::Access::Read);bgfx::setBuffer(1,latched_,bgfx::Access::ReadWrite);bgfx::dispatch(view,latch_,1);
    }
    bgfx::setUniform(params_,p);bgfx::setBuffer(0,states_[current_],bgfx::Access::Read);bgfx::setBuffer(1,current,bgfx::Access::Read);bgfx::setBuffer(2,latched_,bgfx::Access::Read);bgfx::setBuffer(3,states_[1-current_],bgfx::Access::Write);bgfx::dispatch(view,apply_,dispatchGroups(count));current_=1-current_;++ticks_;
}
void VoxelProbes::prepare(std::uint16_t view) {
    const float p[]{0,float(count),float(materials_),0};bgfx::setUniform(sim_,p);
    bgfx::setBuffer(0,states_[current_],bgfx::Access::Read);bgfx::setBuffer(1,renderCopy_,bgfx::Access::Write);bgfx::setBuffer(2,instances_,bgfx::Access::Write);bgfx::setBuffer(3,materialBuffer_,bgfx::Access::Read);bgfx::dispatch(view,pack_,dispatchGroups(count));
}
void VoxelProbes::render(std::uint16_t view) {
    bgfx::setVertexBuffer(0,vertices_);bgfx::setIndexBuffer(indices_);bgfx::setInstanceDataBuffer(instances_,0,count);
    bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);bgfx::submit(view,draw_);
}
}
