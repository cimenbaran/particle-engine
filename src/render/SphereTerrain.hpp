#pragma once
#include "Shaders.hpp"
#include "core/AnalyticSphere.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
namespace pw {
// Fixed unit sphere mesh; the same analytic profile supplies render and physics geometry.
class SphereTerrain {
    bgfx::VertexBufferHandle vertices_=BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle sphere_=BGFX_INVALID_HANDLE;
    void release(){if(bgfx::isValid(vertices_))bgfx::destroy(vertices_);if(bgfx::isValid(indices_))bgfx::destroy(indices_);if(bgfx::isValid(program_))bgfx::destroy(program_);if(bgfx::isValid(sphere_))bgfx::destroy(sphere_);}
public:
    static constexpr std::uint64_t payloadBytes=49ull*97*3*sizeof(float)+48ull*96*6*sizeof(std::uint16_t);
    explicit SphereTerrain(const std::filesystem::path& assets){
        try {
            constexpr unsigned rings=48,sectors=96;std::vector<float> vertices;std::vector<std::uint16_t> indices;
            for(unsigned y=0;y<=rings;++y)for(unsigned x=0;x<=sectors;++x){
                const double latitude=3.14159265358979323846*y/rings,longitude=6.28318530717958647692*x/sectors;
                vertices.push_back(float(std::sin(latitude)*std::cos(longitude)));vertices.push_back(float(std::cos(latitude)));vertices.push_back(float(std::sin(latitude)*std::sin(longitude)));
            }
            for(unsigned y=0;y<rings;++y)for(unsigned x=0;x<sectors;++x){
                const std::uint16_t a=std::uint16_t(y*(sectors+1)+x),b=std::uint16_t(a+sectors+1);
                for(auto i:{a,b,std::uint16_t(a+1),std::uint16_t(a+1),b,std::uint16_t(b+1)})indices.push_back(i);
            }
            bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::Position,3,bgfx::AttribType::Float).end();
            vertices_=bgfx::createVertexBuffer(bgfx::copy(vertices.data(),std::uint32_t(vertices.size()*sizeof(float))),layout);
            indices_=bgfx::createIndexBuffer(bgfx::copy(indices.data(),std::uint32_t(indices.size()*sizeof(std::uint16_t))));
            program_=loadProgram(assets,"vs_terrain","fs_terrain");sphere_=bgfx::createUniform("u_terrain_sphere",bgfx::UniformType::Vec4);
            if(!bgfx::isValid(vertices_) || !bgfx::isValid(indices_) || !bgfx::isValid(sphere_))throw std::runtime_error("Sphere terrain resources unavailable");
        }catch(...){release();throw;}
    }
    ~SphereTerrain(){release();}
    SphereTerrain(const SphereTerrain&)=delete;SphereTerrain& operator=(const SphereTerrain&)=delete;
    void render(std::uint16_t view,const AnalyticSphere& sphere){
        if(sphere.radius<=0)return;
        const float p[]{sphere.centre[0],sphere.centre[1],sphere.centre[2],sphere.radius};bgfx::setUniform(sphere_,p);
        bgfx::setVertexBuffer(0,vertices_);bgfx::setIndexBuffer(indices_);
        bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LESS|BGFX_STATE_MSAA);
        bgfx::submit(view,program_);
    }
};
}
