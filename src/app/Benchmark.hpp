#pragma once
#include "core/RunOptions.hpp"
#include <bgfx/bgfx.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>
namespace pw {
class Benchmark {
public:
    explicit Benchmark(const RunOptions& options):options_(options){}
    void add(std::uint32_t frame,std::uint64_t tick,double wallMs);
    void collect(const bgfx::Stats& stats);
    std::size_t gpuSamples() const;
    void setCompletedWall(double milliseconds) {completedWallMs_=milliseconds;}
    void write(const std::filesystem::path& assets,int width,int height,bool indirect,
               std::uint32_t alive,std::uint32_t visible,std::uint32_t nonBackgroundPixels,const bgfx::Caps& caps) const;
private:
    struct Sample {std::uint32_t frame;std::uint64_t tick;double wallMs,gpuMs=-1;};
    RunOptions options_;
    double completedWallMs_=0;
    std::vector<Sample> samples_;
    std::unordered_map<std::uint32_t,std::size_t> byFrame_;
};
}
