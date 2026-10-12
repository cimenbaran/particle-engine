#pragma once
#include <bgfx/bgfx.h>
#include <filesystem>
#include <string_view>
#include <vector>
namespace pw {
bgfx::ProgramHandle loadProgram(const std::filesystem::path& assets, std::string_view vertex, std::string_view fragment);
bgfx::ProgramHandle loadCompute(const std::filesystem::path& assets, std::string_view name);
// All targets must outlive the transaction. Stage and commit on the main thread
// before submitting the next frame. Destruction discards uncommitted candidates.
class ShaderReload {
public:
    explicit ShaderReload(std::filesystem::path assets):assets_(std::move(assets)) {}
    ~ShaderReload();
    ShaderReload(const ShaderReload&)=delete;
    ShaderReload& operator=(const ShaderReload&)=delete;
    void compute(bgfx::ProgramHandle& target,std::string_view name);
    void graphics(bgfx::ProgramHandle& target,std::string_view vertex,std::string_view fragment);
    void commit();
private:
    void stage(bgfx::ProgramHandle& target,bgfx::ProgramHandle candidate);
    struct Replacement {bgfx::ProgramHandle* target;bgfx::ProgramHandle candidate;};
    std::filesystem::path assets_;
    std::vector<Replacement> replacements_;
    bool failed_=false;
};
void runShaderReloadTest(const std::filesystem::path& assets);
}
