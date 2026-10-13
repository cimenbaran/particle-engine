#pragma once
#include "GpuVoxelWorld.hpp"
namespace pw {
// Swept prediction/response for solid particles against immutable occupancy.
// Caller applies forces first. Any slot error rolls back the entire pass.
class GpuVoxelMotion {
public:
    GpuVoxelMotion(const std::filesystem::path& assets,std::uint32_t capacity);
    ~GpuVoxelMotion();
    GpuVoxelMotion(const GpuVoxelMotion&)=delete;GpuVoxelMotion& operator=(const GpuVoxelMotion&)=delete;
    void submit(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
                std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
                const terrain::SparseVoxelStore& source,terrain::Chunk origin,float dt,float speedLimit=12,
                float bounceThreshold=.5f,std::uint32_t eventBudget=16);
    // Post-pair-solve velocity constraints. Positions stay bitwise unchanged;
    // only newly inward motion produces friction, with no second restitution.
    void finish(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
                std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
                const terrain::SparseVoxelStore& source,terrain::Chunk origin,float speedLimit=12);
    bgfx::DynamicVertexBufferHandle output() const{return output_;}
    // errors: 1 upstream, 2 invalid state/material/overlap, 4 event exhaustion,
    // 8 arithmetic/containment. Motion words: capped bodies, events, normals.
    // Finish words: capped bodies, changed velocities, contact-shell normals;
    // finish bit 4 is contact capacity exhaustion.
    bgfx::DynamicIndexBufferHandle status() const{return status_;}
    static std::uint64_t payloadBytes(std::uint32_t capacity){return std::uint64_t(capacity?capacity:1)*192+16;}
private:
    void release();
    void run(std::uint16_t view,bgfx::DynamicVertexBufferHandle input,bgfx::DynamicVertexBufferHandle materials,
             std::uint32_t materialCount,bgfx::DynamicIndexBufferHandle upstream,const GpuVoxelWorld& world,
             const terrain::SparseVoxelStore& source,terrain::Chunk origin,float dt,float limit,float threshold,
             std::uint32_t budget,bool finish);
    std::uint32_t capacity_;
    std::array<bgfx::DynamicVertexBufferHandle,2> outputs_{{BGFX_INVALID_HANDLE,BGFX_INVALID_HANDLE}};
    bgfx::DynamicVertexBufferHandle candidate_=BGFX_INVALID_HANDLE,output_=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle status_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle clear_=BGFX_INVALID_HANDLE,motion_=BGFX_INVALID_HANDLE,apply_=BGFX_INVALID_HANDLE,finish_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle params_=BGFX_INVALID_HANDLE,profile_=BGFX_INVALID_HANDLE;
};
void runGpuVoxelFinishTest(const std::filesystem::path& assets);
void runGpuVoxelMotionTest(const std::filesystem::path& assets);
}
