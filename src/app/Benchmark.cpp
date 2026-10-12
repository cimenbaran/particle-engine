#include "core/ComputeDispatch.hpp"
#include "Benchmark.hpp"
#include "core/TimingSummary.hpp"
#include <SDL3/SDL.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
namespace pw {
void Benchmark::add(std::uint32_t frame,std::uint64_t tick,double wallMs) {
    byFrame_[frame]=samples_.size();samples_.push_back({frame,tick,wallMs});
}
void Benchmark::collect(const bgfx::Stats& stats) {
    auto found=byFrame_.find(stats.gpuFrameNum);
    if(found!=byFrame_.end() && stats.gpuTimerFreq>0 && stats.gpuTimeEnd>=stats.gpuTimeBegin && stats.gpuTimeEnd>0)
        samples_[found->second].gpuMs=1000.0*double(stats.gpuTimeEnd-stats.gpuTimeBegin)/double(stats.gpuTimerFreq);
}
std::size_t Benchmark::gpuSamples() const {return std::count_if(samples_.begin(),samples_.end(),[](const auto& s){return s.gpuMs>=0;});}
void Benchmark::write(const std::filesystem::path& assets,int width,int height,bool indirect,
                      std::uint32_t alive,std::uint32_t visible,std::uint32_t nonBackgroundPixels,const bgfx::Caps& caps) const {
    using Json=nlohmann::json;
    if(samples_.size()!=options_.frames) throw std::runtime_error("Benchmark was interrupted; no complete report written");
    std::vector<double> wall,gpu;
    for(const auto& sample:samples_) {wall.push_back(sample.wallMs);if(sample.gpuMs>=0) gpu.push_back(sample.gpuMs);}
    auto summary=[](const std::vector<double>& samples)->Json {
        if(samples.empty()) return nullptr;
        auto s=summarize(samples);
        return {{"samples",samples.size()},{"min_ms",s.minimum},{"median_ms",s.median},{"p95_ms",s.p95},{"max_ms",s.maximum},{"mean_ms",s.mean},{"percentile_method","nearest_rank"}};
    };
    Json manifests=Json::object();
    for(const auto& entry:std::filesystem::directory_iterator(assets/"shaders")) if(entry.path().extension()==".json") {
        std::ifstream file(entry.path());Json manifest;file>>manifest;
        manifests[entry.path().filename().string()]=manifest.at("fingerprint");
    }
    std::ifstream buildFile(assets/"build.json");Json build;buildFile>>build;
    std::ifstream materialFile(assets/"materials"/"default.json");Json materialSource;materialFile>>materialSource;
    Json cpuModel=nullptr;
#if defined(__APPLE__)
    char model[256]{};std::size_t length=sizeof(model);
    if(sysctlbyname("machdep.cpu.brand_string",model,&length,nullptr,0)==0) cpuModel=model;
#endif
    Json report={
        {"build",build},{"material_source",materialSource},{"generated_at_unix_ms",std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()},{"schema_version",1},{"scene","seed42_mixed_solids_gravity_analytic_box"},{"seed",42},
        {"capacity",options_.capacity},{"final_active_count",alive},{"final_visible_count",visible},{"final_image_pixels_different_from_corner",nonBackgroundPixels},
        {"warmup_frames",options_.warmup},{"measured_frames",samples_.size()},
        {"render_target","offscreen_rgba8_d32f"},{"resolution",{{"width",width},{"height",height}}},{"vsync",false},{"gui",false},
        {"timed_diagnostics",false},{"frustum_culling",true},{"indirect",indirect},
        {"ticks_per_frame",1},{"tick_seconds",1.0/60.0},{"workgroup_size",pw::workgroupSize},
        {"backend",bgfx::getRendererName(caps.rendererType)},
        {"machine",{{"platform",SDL_GetPlatform()},{"cpu_model",cpuModel},{"logical_cpus",SDL_GetNumLogicalCPUCores()},{"ram_mb",SDL_GetSystemRAM()},{"gpu_vendor_id",caps.vendorId},{"gpu_device_id",caps.deviceId}}},
        {"particle_buffer_bytes_estimate",std::uint64_t(options_.capacity)*200+32+(indirect?32:0)},
        {"shader_fingerprints",manifests},{"submit_frame_wall",summary(wall)},{"gpu_frame",summary(gpu)},
        {"gpu_sample_coverage",double(gpu.size())/samples_.size()},
        {"completed_wall_ms",completedWallMs_},
        {"completed_mean_frame_ms",completedWallMs_/samples_.size()},
        {"completed_throughput_fps",1000.0*samples_.size()/completedWallMs_},
        {"local_sample_within_16_7ms",gpu.size()>=samples_.size()*.95 && summarize(wall).p95<=16.7 && !gpu.empty() && summarize(gpu).p95<=16.7 && completedWallMs_/samples_.size()<=16.7},
        {"notes",Json::array({"One fixed simulation tick per frame; no real-time catch-up scheduler.","Wall samples include submission and frame backpressure; GPU samples use bgfx frame IDs.","Warmup is GPU-drained before measurement; completed wall duration ends after the final counter readback.","Final image validation is outside completed wall duration. Startup correctness tests cover up to 16384 slots.","Buffer estimate excludes materials, geometry, staging, driver allocations and temporary seed upload.","This local measurement does not certify Windows, reference hardware or a ten-minute soak."})}
    };
    std::filesystem::path prefix=options_.benchmark;
    if(!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
    std::ofstream csv(prefix.string()+".csv");csv<<"sample,bgfx_frame,tick,submit_wall_ms,gpu_ms\n"<<std::setprecision(10);
    for(std::size_t i=0;i<samples_.size();++i) {const auto& s=samples_[i];csv<<i<<','<<s.frame<<','<<s.tick<<','<<s.wallMs<<',';if(s.gpuMs>=0) csv<<s.gpuMs;csv<<'\n';}
    csv.close();if(!csv) throw std::runtime_error("Cannot write benchmark CSV");
    std::ofstream json(prefix.string()+".json");json<<report.dump(2)<<'\n';json.close();
    if(!json) throw std::runtime_error("Cannot write benchmark JSON");
    std::cout<<"Benchmark wall p95: "<<summarize(wall).p95<<" ms; GPU samples: "<<gpu.size()<<'/'<<samples_.size()<<"; report: "<<prefix.string()<<".json\n";
}
}
