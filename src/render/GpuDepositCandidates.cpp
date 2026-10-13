#include "GpuDepositCandidates.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <bit>
namespace pw {
GpuDepositCandidates::GpuDepositCandidates(const std::filesystem::path& assets,std::uint32_t capacity,std::span<const terrain::DepositMaterial> materials,double tolerance)
 :capacity_(capacity),materials_(unsigned(materials.size())){
 if(capacity>4096 || materials.empty() || materials.size()>256)throw std::invalid_argument("Invalid deposit candidate bounds");
 terrain::mass_detail::tolerance(tolerance);
 std::vector<std::array<std::uint32_t,4>> profiles;
 for(const auto& m:materials){const auto balance=terrain::planGrainDeposit(2,m.density,tolerance);profiles.push_back({std::bit_cast<std::uint32_t>(balance.inverseGrainMass),unsigned(m.solid),unsigned(balance.admissible && balance.voxels==1),0});}
 try{
  bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
  candidates_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
  profiles_=bgfx::createDynamicVertexBuffer(bgfx::copy(profiles.data(),unsigned(profiles.size()*16)),lanes,BGFX_BUFFER_COMPUTE_READ);
  status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
  clear_=loadCompute(assets,"cs_blast_clear");prepare_=loadCompute(assets,"cs_deposit_candidates");params_=bgfx::createUniform("u_deposit",bgfx::UniformType::Vec4);
  if(!bgfx::isValid(candidates_) || !bgfx::isValid(profiles_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))throw std::runtime_error("Deposit candidate allocation failed");
 }catch(...){release();throw;}
}
void GpuDepositCandidates::release(){for(auto h:{candidates_,profiles_})if(bgfx::isValid(h))bgfx::destroy(h);if(bgfx::isValid(status_))bgfx::destroy(status_);for(auto h:{clear_,prepare_})if(bgfx::isValid(h))bgfx::destroy(h);if(bgfx::isValid(params_))bgfx::destroy(params_);}
GpuDepositCandidates::~GpuDepositCandidates(){release();}
void GpuDepositCandidates::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle particles,bgfx::DynamicVertexBufferHandle history,const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source){
 if(view>=bgfx::getCaps()->limits.maxViews || !world.isCurrent(source,{}) || !bgfx::isValid(particles) || !bgfx::isValid(history) || particles.idx==history.idx)
  throw std::invalid_argument("Invalid deposit source, view or terrain epoch");
 for(auto h:{particles,history})if(h.idx==candidates_.idx || h.idx==profiles_.idx || h.idx==world.buffer().idx)throw std::invalid_argument("Deposit buffer alias");
 bgfx::setViewMode(view,bgfx::ViewMode::Sequential);bgfx::setBuffer(0,status_,bgfx::Access::Write);bgfx::dispatch(view,clear_,1);
 if(!capacity_)return;
 const float p[]{float(capacity_),float(materials_),0,0};bgfx::setUniform(params_,p);
 bgfx::setBuffer(0,particles,bgfx::Access::Read);bgfx::setBuffer(1,history,bgfx::Access::Read);bgfx::setBuffer(2,profiles_,bgfx::Access::Read);bgfx::setBuffer(3,world.buffer(),bgfx::Access::Read);bgfx::setBuffer(4,candidates_,bgfx::Access::Write);bgfx::setBuffer(5,status_,bgfx::Access::ReadWrite);bgfx::dispatch(view,prepare_,dispatchGroups(capacity_));
}
}
