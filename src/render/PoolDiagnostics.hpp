#pragma once
#include <bgfx/bgfx.h>
#include <array>
#include <cstdint>
#include <filesystem>
namespace pw {
class ShaderReload;
struct PoolSnapshot {
    std::uint32_t active{},free{},reserved{},rejected{};
    std::uint64_t tick{},sequence{};
    bool valid=false;
};
// Only 16 bytes per sample. In-flight storage survives scene resets and is never reused early.
class PoolDiagnostics {
public:
    explicit PoolDiagnostics(const std::filesystem::path& assets);
    ~PoolDiagnostics();
    PoolDiagnostics(const PoolDiagnostics&)=delete;
    PoolDiagnostics& operator=(const PoolDiagnostics&)=delete;
    // Schedule after simulation and before any copies; copy uses view+1.
    // The app requests at most one sample per frame. A full ring returns false.
    bool request(std::uint16_t view,bgfx::DynamicIndexBufferHandle counts,std::uint64_t tick);
    void poll(std::uint32_t completedFrame);
    void beginEpoch();
    const PoolSnapshot& latest() const {return latest_;}
    unsigned pending() const;
    void stageShaders(ShaderReload& reload);
private:
    struct Slot {
        bgfx::TextureHandle staging=BGFX_INVALID_HANDLE;
        std::array<std::uint32_t,4> data{};
        std::uint32_t ready{};
        std::uint64_t epoch{},tick{},sequence{};
        bool busy=false;
    };
    void release();
    std::array<Slot,3> slots_;
    bgfx::TextureHandle output_=BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle program_=BGFX_INVALID_HANDLE;
    PoolSnapshot latest_;
    std::uint64_t epoch_{},sequence_{};
};
}
