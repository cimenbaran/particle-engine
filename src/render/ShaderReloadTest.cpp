#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "PoolDiagnostics.hpp"
#include <SDL3/SDL.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace pw {
namespace {
struct Fixture {
    std::filesystem::path assets;
    bgfx::ProgramHandle clear=BGFX_INVALID_HANDLE,draw=BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle counts=BGFX_INVALID_HANDLE;
    bgfx::UniformHandle config=BGFX_INVALID_HANDLE;
    explicit Fixture(const std::filesystem::path& source) {
        assets=std::filesystem::temp_directory_path()/("particleworld-shader-reload-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        try {
            std::filesystem::create_directories(assets);
            std::filesystem::copy(source/"shaders",assets/"shaders",std::filesystem::copy_options::recursive);
            clear=loadCompute(assets,"cs_pool_clear");draw=loadProgram(assets,"vs_particles","fs_particles");
            counts=bgfx::createDynamicIndexBuffer(4,BGFX_BUFFER_INDEX32|BGFX_BUFFER_COMPUTE_READ_WRITE);
            config=bgfx::createUniform("u_pool",bgfx::UniformType::Vec4);
            if(!bgfx::isValid(counts) || !bgfx::isValid(config)) throw std::runtime_error("Reload fixture resources unavailable");
        } catch(...) {release();throw;}
    }
    void release() {
        if(bgfx::isValid(clear)) bgfx::destroy(clear);
        if(bgfx::isValid(draw)) bgfx::destroy(draw);
        if(bgfx::isValid(counts)) bgfx::destroy(counts);
        if(bgfx::isValid(config)) bgfx::destroy(config);
        std::error_code ignored;std::filesystem::remove_all(assets,ignored);
    }
    ~Fixture(){release();}
};
}
void runShaderReloadTest(const std::filesystem::path& assets) {
    Fixture f(assets);
    const auto originalClear=f.clear.idx,originalDraw=f.draw.idx;
    auto reject=[&](auto action) {
        bool failed=false;
        try {ShaderReload reload(f.assets);reload.compute(f.clear,"cs_reset");action(reload);reload.commit();}
        catch(const std::exception&) {failed=true;}
        if(!failed || f.clear.idx!=originalClear || f.draw.idx!=originalDraw)
            throw std::runtime_error("Shader reload did not retain the complete original set");
        bgfx::frame();
    };
    reject([&](auto& reload){reload.graphics(f.draw,"vs_particles","missing_fragment");});
    const auto fs=f.assets/"shaders/fs_particles.bin";
    {std::ofstream corrupt(fs,std::ios::binary|std::ios::app);corrupt<<'x';}
    reject([&](auto& reload){reload.graphics(f.draw,"vs_particles","fs_particles");});
    std::filesystem::copy_file(assets/"shaders/fs_particles.bin",fs,std::filesystem::copy_options::overwrite_existing);
    {std::ofstream corrupt(fs.string()+".json");corrupt<<"{malformed";}
    reject([&](auto& reload){reload.graphics(f.draw,"vs_particles","fs_particles");});
    std::filesystem::copy_file(assets/"shaders/fs_particles.bin.json",fs.string()+".json",std::filesystem::copy_options::overwrite_existing);
    {
        std::ifstream input(fs.string()+".json");nlohmann::json metadata;input>>metadata;input.close();
        metadata["platform"]="wrong_backend";
        std::ofstream output(fs.string()+".json");output<<metadata;output.close();
        reject([&](auto& reload){reload.graphics(f.draw,"vs_particles","fs_particles");});
        std::filesystem::copy_file(assets/"shaders/fs_particles.bin.json",fs.string()+".json",std::filesystem::copy_options::overwrite_existing);
    }
    {
        ShaderReload reload(f.assets);reload.compute(f.clear,"cs_reset");
        try {reload.compute(f.clear,"missing_compute");} catch(const std::exception&) {}
        bool refused=false;try {reload.commit();} catch(const std::exception&) {refused=true;}
        if(!refused || f.clear.idx!=originalClear) throw std::runtime_error("Failed reload was partially committed");
    }
    {
        const auto path=f.assets/"shaders/cs_pool_clear.bin.json";
        std::ifstream input(path);nlohmann::json metadata;input>>metadata;input.close();
        metadata["defines"]="PW_WORKGROUP_SIZE="+std::to_string(workgroupSize==128?256:128);
        std::ofstream output(path);output<<metadata;output.close();
        reject([&](auto& reload){reload.compute(f.draw,"cs_pool_clear");});
        std::filesystem::copy_file(assets/"shaders/cs_pool_clear.bin.json",path,std::filesystem::copy_options::overwrite_existing);
    }
    reject([&](auto& reload){reload.compute(f.clear,"cs_pool_clear");}); // duplicate target
    bgfx::destroy(f.clear);f.clear=loadCompute(f.assets,"cs_reset");
    for(unsigned i=0;i<8;++i) {
        ShaderReload reload(f.assets);reload.compute(f.clear,"cs_pool_clear");
        reload.graphics(f.draw,"vs_particles","fs_particles");reload.commit();
        const float config[]{4,0,0,1};bgfx::setUniform(f.config,config);
        bgfx::setBuffer(0,f.counts,bgfx::Access::Write);bgfx::dispatch(0,f.clear,1);
        bgfx::frame();
    }
    // Read back a real dispatch after failures and repeated same-handle swaps.
    PoolDiagnostics diagnostics(assets);
    diagnostics.request(1,f.counts,0);
    while(diagnostics.pending()) {diagnostics.poll(bgfx::frame());SDL_PumpEvents();}
    const auto& data=diagnostics.latest();
    if(!data.valid || data.active || data.free || data.reserved || data.rejected)
        throw std::runtime_error("Reloaded compute program failed execution");
    std::cout<<"Shader reload passed: missing/corrupt candidates, atomic rollback, duplicate targets and repeated swaps with GPU execution\n";
}
}
