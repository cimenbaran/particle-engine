#pragma once
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
namespace pw {
struct RunOptions {
    std::uint32_t capacity=16384,cosmeticCapacity=16384,frames=0,warmup=120,soakSeconds=0;
    bool gpuTest=false,indirect=true,softwareGpu=false,granular=false,granularCheck=false;
    std::string benchmark,report,capabilities,scene,replay;
};
inline RunOptions parseOptions(int argc,char** argv) {
    RunOptions options;bool specifiedWarmup=false,specifiedCapacity=false,specifiedCosmetics=false;
    auto value=[&](int& index) -> std::string_view {
        if(++index>=argc) throw std::runtime_error("Missing option value");return argv[index];
    };
    auto number=[](std::string_view text,std::uint32_t low,std::uint32_t high) {
        std::uint32_t result{};auto parsed=std::from_chars(text.data(),text.data()+text.size(),result);
        if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || result<low || result>high)
            throw std::runtime_error("Numeric option is outside its supported range");
        return result;
    };
    for(int i=1;i<argc;++i) {
        const std::string_view arg=argv[i];
        if(arg=="--capacity") {options.capacity=number(value(i),1,1048576);specifiedCapacity=true;}
        else if(arg=="--cosmetic-capacity") {options.cosmeticCapacity=number(value(i),0,150000);specifiedCosmetics=true;}
        else if(arg=="--frames") options.frames=number(value(i),1,1000000);
        else if(arg=="--warmup") {options.warmup=number(value(i),0,100000);specifiedWarmup=true;}
        else if(arg=="--benchmark") {options.benchmark=value(i);if(options.benchmark.empty()) throw std::runtime_error("Empty benchmark prefix");}
        else if(arg=="--soak") options.soakSeconds=number(value(i),1,3600);
        else if(arg=="--report") {options.report=value(i);if(options.report.empty()) throw std::runtime_error("Empty report path");}
        else if(arg=="--capabilities") {options.capabilities=value(i);if(options.capabilities.empty()) throw std::runtime_error("Empty capability report path");}
        else if(arg=="--scene") {options.scene=value(i);if(options.scene.empty()) throw std::runtime_error("Empty scene path");}
        else if(arg=="--replay") {options.replay=value(i);if(options.replay.empty()) throw std::runtime_error("Empty replay path");}
        else if(arg=="--granular-check") options.granularCheck=true;
        else if(arg=="--granular") options.granular=true;
        else if(arg=="--no-indirect") options.indirect=false;
        else if(arg=="--gpu-test") options.gpuTest=true;
        else if(arg=="--software-gpu") options.softwareGpu=true;
        else throw std::runtime_error("Usage: ParticleWorld [--capacity 1..1048576] [--cosmetic-capacity 0..150000] [--frames N] [--granular] [--granular-check --report path.json] [--gpu-test] [--software-gpu] [--no-indirect] [--benchmark output-prefix --warmup N] [--soak seconds --report path.json] [--capabilities path.json] [--scene path.json] [--replay recording.json --report result.json]");
    }
    if(options.granularCheck && (options.report.empty() || options.granular || options.gpuTest || options.soakSeconds ||
       !options.benchmark.empty() || !options.replay.empty() || !options.capabilities.empty() || !options.scene.empty() ||
       options.frames || specifiedWarmup || specifiedCapacity || specifiedCosmetics))
        throw std::runtime_error("--granular-check requires --report and uses fixed validation fixtures");
    if(options.granular) {
        if(options.gpuTest || !options.benchmark.empty() || !options.replay.empty() || !options.capabilities.empty())
            throw std::runtime_error("--granular supports interactive, bounded-frame and soak runs");
        if(!specifiedCapacity) options.capacity=512;
        if(!specifiedCosmetics) options.cosmeticCapacity=0;
    }
    if(options.softwareGpu && !options.gpuTest && !options.granularCheck && !(options.granular && options.soakSeconds) && options.capabilities.empty() && options.replay.empty())
        throw std::runtime_error("--software-gpu is restricted to Windows GPU tests, granular checks/soaks, replay or capability reporting");
    if(specifiedCosmetics && (options.gpuTest || options.soakSeconds || !options.replay.empty() || !options.capabilities.empty()))
        throw std::runtime_error("--cosmetic-capacity is for interactive, bounded-frame or benchmark runs only");
    if(!options.replay.empty() && (options.gpuTest || options.soakSeconds || !options.benchmark.empty() ||
       !options.capabilities.empty() || !options.scene.empty() || options.frames || specifiedWarmup || specifiedCapacity))
        throw std::runtime_error("--replay uses recorded capacity/ticks and cannot combine with other execution modes");
    if(!options.scene.empty() && (options.gpuTest || (options.soakSeconds && !options.granular) || !options.benchmark.empty() || !options.capabilities.empty()))
        throw std::runtime_error("--scene supports interactive, bounded-frame or explicit granular soak runs");
    if(!options.capabilities.empty() && (options.soakSeconds || options.gpuTest || !options.benchmark.empty() || options.frames || specifiedWarmup || !options.report.empty()))
        throw std::runtime_error("--capabilities cannot combine with execution modes or timing/report options");
    if(options.soakSeconds && (options.report.empty() || options.gpuTest || !options.benchmark.empty() || options.frames || specifiedWarmup))
        throw std::runtime_error("--soak requires --report and cannot combine with GPU-test, benchmark, frames or warmup modes");
    if(!options.report.empty() && !options.soakSeconds && options.replay.empty() && !options.granularCheck) throw std::runtime_error("--report requires --soak, --replay or --granular-check");
    if(options.gpuTest && !options.benchmark.empty()) throw std::runtime_error("GPU tests and benchmark modes are mutually exclusive");
    if(specifiedWarmup && options.benchmark.empty()) throw std::runtime_error("--warmup requires --benchmark");
    if(!options.benchmark.empty()) {
        if(!options.frames) options.frames=600;
        if(!specifiedCosmetics) options.cosmeticCapacity=0;
    }
    return options;
}
}
