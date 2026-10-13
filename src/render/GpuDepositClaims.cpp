#include "GpuDepositClaims.hpp"
#include "Shaders.hpp"
namespace pw {
GpuDepositClaims::GpuDepositClaims(const std::filesystem::path& assets,std::uint32_t capacity,std::span<const terrain::DepositMaterial> materials,std::uint32_t quota,double tolerance)
 :candidates_(assets,capacity,materials,tolerance),capacity_(capacity),quota_(quota){
 if(quota>512)throw std::invalid_argument("Deposit claim quota exceeds 512");
 try{
  bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
  claims_=bgfx::createDynamicVertexBuffer(std::max(1u,quota)*2,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
  owners_=bgfx::createDynamicIndexBuffer(std::max(1u,capacity),BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
  status_=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
  program_=loadCompute(assets,"cs_deposit_claims");params_=bgfx::createUniform("u_deposit_claims",bgfx::UniformType::Vec4);
  if(!bgfx::isValid(claims_) || !bgfx::isValid(owners_) || !bgfx::isValid(status_) || !bgfx::isValid(params_))throw std::runtime_error("Deposit claim allocation failed");
 }catch(...){release();throw;}
}
void GpuDepositClaims::release(){if(bgfx::isValid(claims_))bgfx::destroy(claims_);for(auto h:{owners_,status_})if(bgfx::isValid(h))bgfx::destroy(h);if(bgfx::isValid(program_))bgfx::destroy(program_);if(bgfx::isValid(params_))bgfx::destroy(params_);}
GpuDepositClaims::~GpuDepositClaims(){release();}
void GpuDepositClaims::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle particles,bgfx::DynamicVertexBufferHandle history,const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source){
 if(particles.idx==claims_.idx || history.idx==claims_.idx)throw std::invalid_argument("Deposit claim output alias");
 candidates_.submit(view,particles,history,world,source);
 const float p[]{float(capacity_),float(quota_),0,0};bgfx::setUniform(params_,p);
 bgfx::setBuffer(0,particles,bgfx::Access::Read);bgfx::setBuffer(1,candidates_.output(),bgfx::Access::Read);bgfx::setBuffer(2,candidates_.status(),bgfx::Access::Read);bgfx::setBuffer(3,owners_,bgfx::Access::ReadWrite);bgfx::setBuffer(4,claims_,bgfx::Access::ReadWrite);bgfx::setBuffer(5,status_,bgfx::Access::Write);bgfx::dispatch(view,program_,1);
}
}
