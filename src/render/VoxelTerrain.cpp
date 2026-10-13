#include "VoxelTerrain.hpp"
#include "Shaders.hpp"
#include "core/VoxelSurface.hpp"
#include <algorithm>
#include <cmath>
namespace pw {
namespace {
struct TerrainVertex {float x,y,z;std::uint32_t color;};
static_assert(sizeof(TerrainVertex)==16);
}
VoxelTerrain::VoxelTerrain(const std::filesystem::path& assets,const terrain::SparseVoxelStore& source,
                           const GpuVoxelWorld& world,terrain::Chunk origin,const MaterialLibrary& materials)
    :owner_(&source),origin_(origin),revision_(source.revision()) {
    if(!world.isCurrent(source,origin))throw std::invalid_argument("Terrain mesh needs current occupancy/origin");
    std::vector<TerrainVertex> vertices;std::vector<std::uint32_t> indices;
    for(unsigned slot=0;slot<source.layout().chunkCount();++slot) {
        const auto chunk=*source.layout().decode(slot);const auto surface=*terrain::buildVoxelSurface(source,chunk);
        if(surface.faces.size()>1048576-faces_)throw std::runtime_error("Terrain surface exceeds one million face budget");
        const auto offset=*terrain::relativeCorner({chunk,{}},origin);
        for(const auto& face:surface.faces) {
            if(face.material>=materials.entries.size())throw std::invalid_argument("Terrain material outside library");
            const auto& material=materials.entries[face.material].gpu;
            if(material.identity[1]!=0)throw std::invalid_argument("Opaque voxel surface requires solid material");
            constexpr float light[3]{-.39036003f,.78072006f,.48795004f};
            const float shade=.3f+.7f*std::max(0.0f,light[face.side/2]*((face.side&1)?1.0f:-1.0f));
            std::uint32_t color=0xff000000u;
            for(unsigned channel=0;channel<3;++channel)color|=std::uint32_t(std::lround(std::clamp(material.color[channel]*shade,0.0f,1.0f)*255))<<(channel*8);
            const auto first=std::uint32_t(vertices.size());
            for(const auto& corner:*terrain::faceCorners(face))
                vertices.push_back({float(offset[0]+corner[0]*.125),float(offset[1]+corner[1]*.125),float(offset[2]+corner[2]*.125),color});
            for(unsigned index:{0,1,2,0,2,3})indices.push_back(first+index);
            ++faces_;
        }
    }
    try {
        program_=loadProgram(assets,"vs_voxel_terrain","fs_terrain");
        if(faces_) {
            bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::Position,3,bgfx::AttribType::Float).add(bgfx::Attrib::Color0,4,bgfx::AttribType::Uint8,true).end();
            vertices_=bgfx::createVertexBuffer(bgfx::copy(vertices.data(),std::uint32_t(vertices.size()*sizeof(TerrainVertex))),layout);
            indices_=bgfx::createIndexBuffer(bgfx::copy(indices.data(),std::uint32_t(indices.size()*sizeof(std::uint32_t))),BGFX_BUFFER_INDEX32);
            if(!bgfx::isValid(vertices_) || !bgfx::isValid(indices_))throw std::runtime_error("Terrain mesh allocation failed");
        }
    }catch(...){release();throw;}
}
VoxelTerrain::~VoxelTerrain(){release();}
void VoxelTerrain::release(){if(bgfx::isValid(vertices_))bgfx::destroy(vertices_);if(bgfx::isValid(indices_))bgfx::destroy(indices_);if(bgfx::isValid(program_))bgfx::destroy(program_);}
void VoxelTerrain::render(std::uint16_t view,const terrain::SparseVoxelStore& source,const GpuVoxelWorld& world,terrain::Chunk origin) const {
    if(owner_!=&source || origin_!=origin || revision_!=world.revision() || !world.isCurrent(source,origin))
        throw std::invalid_argument("Stale terrain mesh/occupancy publication");
    if(!faces_)return;
    bgfx::setVertexBuffer(0,vertices_);bgfx::setIndexBuffer(indices_);
    bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);
    bgfx::submit(view,program_);
}
}
