#pragma once
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
namespace pw {
struct RunOptions {
    std::uint32_t capacity=16384,frames=0,warmup=120,soakSeconds=0;
    bool gpuTest=false,indirect=true;
    std::string benchmark,report;
};
inline RunOptions parseOptions(int argc,char** argv) {
    RunOptions options;bool specifiedWarmup=false;
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
        if(arg=="--capacity") options.capacity=number(value(i),1,1048576);
        else if(arg=="--frames") options.frames=number(value(i),1,1000000);
        else if(arg=="--warmup") {options.warmup=number(value(i),0,100000);specifiedWarmup=true;}
        else if(arg=="--benchmark") {options.benchmark=value(i);if(options.benchmark.empty()) throw std::runtime_error("Empty benchmark prefix");}
        else if(arg=="--soak") options.soakSeconds=number(value(i),1,3600);
        else if(arg=="--report") {options.report=value(i);if(options.report.empty()) throw std::runtime_error("Empty soak report path");}
        else if(arg=="--no-indirect") options.indirect=false;
        else if(arg=="--gpu-test") options.gpuTest=true;
        else throw std::runtime_error("Usage: ParticleWorld [--capacity 1..1048576] [--frames N] [--gpu-test] [--no-indirect] [--benchmark output-prefix --warmup N] [--soak seconds --report path.json]");
    }
    if(options.soakSeconds && (options.report.empty() || options.gpuTest || !options.benchmark.empty() || options.frames || specifiedWarmup))
        throw std::runtime_error("--soak requires --report and cannot combine with GPU-test, benchmark, frames or warmup modes");
    if(!options.report.empty() && !options.soakSeconds) throw std::runtime_error("--report requires --soak");
    if(options.gpuTest && !options.benchmark.empty()) throw std::runtime_error("GPU tests and benchmark modes are mutually exclusive");
    if(specifiedWarmup && options.benchmark.empty()) throw std::runtime_error("--warmup requires --benchmark");
    if(!options.benchmark.empty() && !options.frames) options.frames=600;
    return options;
}
}
