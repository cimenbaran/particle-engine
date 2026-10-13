#include "GpuVoxelWorld.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <bit>
#include <vector>
namespace pw {
GpuVoxelWorld::GpuVoxelWorld(const terrain::SparseVoxelStore& source,terrain::Chunk origin)
    :owner_(&source),origin_(origin),revision_(source.revision()),bytes_(64+4*source.layout().voxelCount()) {
    const auto& layout=source.layout();
    if(layout.chunkCount()>256)throw std::invalid_argument("GPU resident terrain exceeds 256 chunks");
    const auto minimum=layout.minimum();const auto dims=layout.dimensions();
    std::array<std::int32_t,3> relative;
    for(unsigned axis=0;axis<3;++axis) {
        const auto lower=(std::int64_t(minimum[axis])-origin[axis])*32;
        const auto upper=lower+std::int64_t(dims[axis])*32;
        if(lower< -512 || upper>512)throw std::invalid_argument("GPU terrain exceeds shared origin range");
        relative[axis]=std::int32_t(lower);
    }
    std::vector<std::uint32_t> words(std::size_t(bytes_/4),terrain::SparseVoxelStore::empty);
    for(unsigned axis=0;axis<3;++axis){words[axis]=std::bit_cast<std::uint32_t>(relative[axis]);words[4+axis]=dims[axis];words[12+axis]=std::bit_cast<std::uint32_t>(origin[axis]);}
    words[3]=layout.chunkCount();words[7]=std::uint32_t(layout.voxelCount());
    words[8]=std::uint32_t(revision_);words[9]=std::uint32_t(revision_>>32);words[10]=words[11]=words[15]=0;
    for(std::uint32_t slot=0;slot<layout.chunkCount();++slot)
        source.copyChunk(*layout.decode(slot),std::span<std::uint32_t,terrain::voxelsPerChunk>(words.data()+16+slot*terrain::voxelsPerChunk,terrain::voxelsPerChunk));
    bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    buffer_=bgfx::createDynamicVertexBuffer(bgfx::copy(words.data(),std::uint32_t(bytes_)),lanes,BGFX_BUFFER_COMPUTE_READ);
    if(!bgfx::isValid(buffer_))throw std::runtime_error("GPU terrain allocation failed");
}
GpuVoxelWorld::~GpuVoxelWorld(){if(bgfx::isValid(buffer_))bgfx::destroy(buffer_);}
bool GpuVoxelWorld::isCurrent(const terrain::SparseVoxelStore& source,terrain::Chunk origin) const {
    return owner_==&source && origin_==origin && revision_==source.revision();
}
GpuVoxelWorldSweep::GpuVoxelWorldSweep(const std::filesystem::path& assets,std::uint32_t capacity):capacity_(capacity) {
    if(capacity>1048576)throw std::invalid_argument("World query capacity exceeds limit");
    try {
        bgfx::VertexLayout lanes;lanes.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
        output_=bgfx::createDynamicVertexBuffer(std::max(1u,capacity)*3,lanes,BGFX_BUFFER_COMPUTE_READ_WRITE);
        program_=loadCompute(assets,"cs_voxel_world_sweep");params_=bgfx::createUniform("u_voxel_world_query",bgfx::UniformType::Vec4);
        if(!bgfx::isValid(output_) || !bgfx::isValid(program_) || !bgfx::isValid(params_))throw std::runtime_error("World query allocation failed");
    }catch(...){release();throw;}
}
GpuVoxelWorldSweep::~GpuVoxelWorldSweep(){release();}
void GpuVoxelWorldSweep::release(){
    if(bgfx::isValid(output_))bgfx::destroy(output_);
    if(bgfx::isValid(program_))bgfx::destroy(program_);
    if(bgfx::isValid(params_))bgfx::destroy(params_);
}
void GpuVoxelWorldSweep::submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle segments,
                              const GpuVoxelWorld& world,const terrain::SparseVoxelStore& source,terrain::Chunk origin) {
    if(!world.isCurrent(source,origin))throw std::invalid_argument("Stale or mismatched GPU terrain snapshot");
    if(!bgfx::isValid(segments) || segments.idx==output_.idx || segments.idx==world.buffer().idx)
        throw std::invalid_argument("Invalid world query segments");
    if(!capacity_)return;
    bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
    const float p[]{float(capacity_),0,0,0};bgfx::setUniform(params_,p);
    bgfx::setBuffer(0,segments,bgfx::Access::Read);bgfx::setBuffer(1,output_,bgfx::Access::Write);bgfx::setBuffer(2,world.buffer(),bgfx::Access::Read);
    bgfx::dispatch(view,program_,dispatchGroups(capacity_));
}
}
