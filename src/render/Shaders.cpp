#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <array>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#else
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#endif
namespace pw {
static std::string digest(const std::vector<char>& data) {
    std::array<unsigned char,32> hash{};
#if defined(__APPLE__)
    CC_SHA256(data.data(),CC_LONG(data.size()),hash.data());
#else
    if(BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,
        reinterpret_cast<PUCHAR>(const_cast<char*>(data.data())),ULONG(data.size()),hash.data(),ULONG(hash.size()))<0)
        throw std::runtime_error("Shader SHA-256 failed");
#endif
    constexpr char hex[]="0123456789abcdef";
    std::string result;result.reserve(64);
    for(auto byte:hash) {result+=hex[byte>>4];result+=hex[byte&15];}
    return result;
}
static bgfx::ShaderHandle shader(const std::filesystem::path& assets, std::string_view name,char stage) {
    auto path = assets / "shaders" / (std::string(name) + ".bin");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader: " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || size > 16 * 1024 * 1024) throw std::runtime_error("Invalid shader size");
    std::vector<char> data(static_cast<std::size_t>(size));
    file.seekg(0); file.read(data.data(), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Cannot read shader: " + path.string());
    std::ifstream metadata(path.string()+".json");
    nlohmann::json manifest;
    if(!metadata || !(metadata>>manifest)) throw std::runtime_error("Cannot read shader manifest: "+path.string());
    const auto backend=bgfx::getRendererType();
    const std::string platform=backend==bgfx::RendererType::Metal?"osx":"windows";
    const std::string profile=backend==bgfx::RendererType::Metal?"metal":"s_5_0";
    const std::string type=stage=='C'?"compute":stage=='V'?"vertex":"fragment";
    if((manifest.at("schema")!=1 && manifest.at("schema")!=2) || manifest.at("platform")!=platform ||
       manifest.at("profile")!=profile || manifest.at("type")!=type ||
       manifest.at("shader")!=std::string(name)+".sc" || manifest.at("binary_sha256")!=digest(data))
        throw std::runtime_error("Shader manifest or binary mismatch: "+path.string());
    if(stage=='C' && manifest.value("defines",std::string{})!="PW_WORKGROUP_SIZE="+std::to_string(workgroupSize))
        throw std::runtime_error("Shader workgroup variant does not match this executable: "+path.string());
    if(data.size()<22 || data[0]!=stage || data[1]!='S' || data[2]!='H' || data[3]!=12)
        throw std::runtime_error("Shader header incompatible with pinned bgfx: "+path.string());
    auto handle = bgfx::createShader(bgfx::copy(data.data(), static_cast<std::uint32_t>(data.size())));
    if (!bgfx::isValid(handle)) throw std::runtime_error("Shader rejected: " + path.string());
    return handle;
}
bgfx::ProgramHandle loadCompute(const std::filesystem::path& assets, std::string_view name) {
    auto cs=shader(assets,name,'C');
    auto program=bgfx::createProgram(cs,false);
    bgfx::destroy(cs);
    if (!bgfx::isValid(program)) throw std::runtime_error("Compute program creation failed");
    return program;
}
bgfx::ProgramHandle loadProgram(const std::filesystem::path& assets, std::string_view vertex, std::string_view fragment) {
    auto vs = shader(assets, vertex,'V');
    bgfx::ShaderHandle fs = BGFX_INVALID_HANDLE;
    try { fs = shader(assets, fragment,'F'); } catch (...) { bgfx::destroy(vs); throw; }
    auto program = bgfx::createProgram(vs, fs, false);
    bgfx::destroy(vs);bgfx::destroy(fs);
    if (!bgfx::isValid(program)) throw std::runtime_error("Graphics program creation failed");
    return program;
}

ShaderReload::~ShaderReload() {
    for(auto& item:replacements_) if(bgfx::isValid(item.candidate)) bgfx::destroy(item.candidate);
}
void ShaderReload::stage(bgfx::ProgramHandle& target,bgfx::ProgramHandle candidate) {
    try {
        for(const auto& item:replacements_) if(item.target==&target) throw std::runtime_error("Duplicate shader reload target");
        replacements_.push_back({&target,candidate});
    } catch(...) {bgfx::destroy(candidate);throw;}
}
void ShaderReload::compute(bgfx::ProgramHandle& target,std::string_view name) {
    try {
        if(failed_) throw std::runtime_error("Shader reload transaction has failed");
        stage(target,loadCompute(assets_,name));
    } catch(...) {failed_=true;throw;}
}
void ShaderReload::graphics(bgfx::ProgramHandle& target,std::string_view vertex,std::string_view fragment) {
    try {
        if(failed_) throw std::runtime_error("Shader reload transaction has failed");
        stage(target,loadProgram(assets_,vertex,fragment));
    } catch(...) {failed_=true;throw;}

}
void ShaderReload::commit() {
    if(failed_) throw std::runtime_error("Cannot commit a failed shader reload transaction");
    for(auto& item:replacements_) {
        if(bgfx::isValid(*item.target)) bgfx::destroy(*item.target);
        *item.target=item.candidate;item.candidate=BGFX_INVALID_HANDLE;
    }
    replacements_.clear();
}
}
