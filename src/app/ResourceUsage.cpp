#include "ResourceUsage.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include <imgui.h>
#include "core/ComputeDispatch.hpp"
#include <algorithm>
#include <bit>
namespace pw {
ResourceUsage& ResourceUsage::operator+=(const ResourceUsage& r) {
    state+=r.state;instances+=r.instances;lists+=r.lists;counters+=r.counters;indirect+=r.indirect;
    materials+=r.materials;geometry+=r.geometry;terrainGeometry+=r.terrainGeometry;diagnosticTextures+=r.diagnosticTextures;
    diagnosticHost+=r.diagnosticHost;auditBuffers+=r.auditBuffers;seedUpload+=r.seedUpload;granular+=r.granular;return *this;
}

ResourceUsage resourceUsage(std::uint32_t capacity,std::uint32_t materialCount,bool indirectAllocated,bool audit) {
    const std::uint64_t n=capacity;
    ResourceUsage r;
    r.state=2*n*sizeof(ParticleGpu);r.instances=2*n*sizeof(RenderInstance);
    r.lists=2*n*sizeof(std::uint32_t);r.counters=2*4*sizeof(std::uint32_t);
    // Pinned bgfx uses an eight-word indirect command, including reserved words.
    r.indirect=indirectAllocated?32:0;r.materials=std::uint64_t(materialCount)*sizeof(MaterialGpu);
    r.geometry=4*3*sizeof(float)+6*sizeof(std::uint16_t);
    const auto diagnosticRings=audit?3u:2u;
    // Each ring has one 1x1 RGBA32U export texture and three readback textures.
    r.diagnosticTextures=diagnosticRings*4*16;r.diagnosticHost=diagnosticRings*3*16;
    r.auditBuffers=audit?(n+4)*sizeof(std::uint32_t):0;
    r.seedUpload=n*sizeof(ParticleGpu);
    return r;
}
std::uint64_t granularPayloadBytes(std::uint32_t capacity) {
    std::uint64_t scan=0;auto count=capacity;
    for(;;){const auto groups=std::max(1u,dispatchGroups(count));scan+=4ull*(std::max(1u,count)+groups);if(groups==1)break;count=groups;}
    // All component payloads + two full indices with four recursive scans.
    return 1312ull*std::max(1u,capacity)+288+4*scan+16ull*std::bit_ceil(capacity*2+2);
}
std::uint64_t settlingPayloadBytes(std::uint32_t capacity) {
    std::uint64_t scan=0;auto count=capacity;
    for(;;){const auto groups=std::max(1u,dispatchGroups(count));scan+=4ull*(std::max(1u,count)+groups);if(groups==1)break;count=groups;}
    return 80ull*std::max(1u,capacity)+96+2*scan+8ull*std::bit_ceil(capacity*2+2);
}
nlohmann::json resourceReport(const ResourceUsage& u,const bgfx::Stats& s) {
    using Json=nlohmann::json;
    auto available=[](std::int64_t bytes)->Json {return bytes<0?Json(nullptr):Json(bytes);};
    return {{"schema_version",1},{"requested_payload_bytes",{
        {"particle_state",u.state},{"granular_solver",u.granular},{"render_instances",u.instances},{"active_free_lists",u.lists},
        {"pool_visibility_counters",u.counters},{"indirect_arguments",u.indirect},{"material_table",u.materials},
        {"billboard_geometry",u.geometry},{"terrain_geometry",u.terrainGeometry},{"diagnostic_textures",u.diagnosticTextures},{"audit_buffers",u.auditBuffers},
        {"persistent_total",u.persistentPayload()},{"particle_buffers",u.particleBuffers()}}},
        {"host_diagnostic_payload_bytes",u.diagnosticHost},{"temporary_seed_upload_payload_bytes",u.seedUpload},
        {"renderer_estimates_bytes",{{"gpu_used",available(s.gpuMemoryUsed)},{"gpu_available",available(s.gpuMemoryMax)},
            {"textures",available(s.textureMemoryUsed)},{"render_targets",available(s.rtMemoryUsed)},
            {"transient_vertices_used",available(s.transientVbUsed)},{"transient_indices_used",available(s.transientIbUsed)}}},
        {"handles",{{"dynamic_vertex_buffers",s.numDynamicVertexBuffers},{"dynamic_index_buffers",s.numDynamicIndexBuffers},
            {"vertex_buffers",s.numVertexBuffers},{"index_buffers",s.numIndexBuffers},{"textures",s.numTextures},
            {"framebuffers",s.numFrameBuffers},{"programs",s.numPrograms},{"shaders",s.numShaders},{"uniforms",s.numUniforms}}},
        {"provisional_budget_bytes",{{"particle_working_memory",512000000},{"total_gpu_allocations",1500000000}}},
        {"notes",Json::array({"Payload is requested data size; excludes alignment, allocator pages, driver copies, shaders and uniforms.",
            "Reset can retain old buffers until deferred destruction; seed upload is temporary and is not a peak-memory estimate.",
            "Host seed vectors, bgfx upload copies, GUI/font resources and render targets are outside the requested persistent total.",
            "Renderer estimates overlap payload categories; do not add them together. Negative/unavailable counters are null.",
            "This snapshot does not certify either provisional memory budget."})}};
}
ReportedMemory reportedMemory(const bgfx::Stats& s) {
    return {s.gpuMemoryUsed,s.textureMemoryUsed,s.rtMemoryUsed,s.transientVbUsed,s.transientIbUsed};
}
nlohmann::json resourcePeakReport(const ReportedMemoryPeaks& peaks,const char* scope) {
    using Json=nlohmann::json;
    auto available=[](std::int64_t value)->Json {return value<0?Json(nullptr):Json(value);};
    const auto& p=peaks.values();
    return {{"schema_version",1},{"scope",scope},{"observations",peaks.observations()},
        {"bytes",{{"gpu_used",available(p.gpu)},{"textures",available(p.textures)},
            {"render_targets",available(p.renderTargets)},{"transient_vertices_used",available(p.transientVertices)},
            {"transient_indices_used",available(p.transientIndices)}}},
        {"note","Maxima of delayed renderer estimates at sampled frames; misses unsampled peaks and does not certify total allocation budgets."}};
}
void drawResourceUsage(const ResourceUsage& u,const bgfx::Stats& s,ReportedMemoryPeaks& peaks) {
    ImGui::SetNextWindowPos(ImVec2(492,248),ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420,420),ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true,ImGuiCond_FirstUseEver);
    ImGui::Begin("Allocations");
    ImGui::TextUnformatted("Requested GPU payload: all pools (MB)");
    if(ImGui::BeginTable("payload",2,ImGuiTableFlags_SizingStretchProp)) {
        auto row=[](const char* name,std::uint64_t bytes) {
            ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(name);
            ImGui::TableNextColumn();ImGui::Text("%.6f",double(bytes)/1000000.0);
        };
        if(u.granular) row("Granular solver",u.granular);
        row("State ping-pong",u.state);row("Render + visible instances",u.instances);
        row("Active/free slot lists",u.lists);row("Pool/visibility counters",u.counters);
        row("Indirect arguments",u.indirect);row("Materials + geometry",u.materials+u.geometry+u.terrainGeometry);
        row("Diagnostic textures",u.diagnosticTextures);if(u.auditBuffers) row("Audit buffers",u.auditBuffers);
        row("Persistent payload total",u.persistentPayload());row("Temporary seed upload",u.seedUpload);
        ImGui::EndTable();
    }
    ImGui::Text("Host readback payload: %llu bytes",static_cast<unsigned long long>(u.diagnosticHost));
    ImGui::Separator();
    auto estimate=[](const char* name,std::int64_t bytes) {
        if(bytes<0) ImGui::Text("%s: unavailable",name);
        else ImGui::Text("%s: %.3f MB",name,double(bytes)/1000000.0);
    };
    estimate("Driver GPU usage",s.gpuMemoryUsed);estimate("Driver available",s.gpuMemoryMax);
    estimate("Renderer textures",s.textureMemoryUsed);estimate("Renderer render targets",s.rtMemoryUsed);
    estimate("Frame transient vertices",s.transientVbUsed);estimate("Frame transient indices",s.transientIbUsed);
    ImGui::Separator();
    if(ImGui::Button("Reset observed peaks")) {peaks.reset();peaks.observe(reportedMemory(s));}
    ImGui::Text("Peak window: %llu observations",static_cast<unsigned long long>(peaks.observations()));
    const auto& peak=peaks.values();
    estimate("Peak reported GPU usage",peak.gpu);estimate("Peak renderer textures",peak.textures);
    estimate("Peak render targets",peak.renderTargets);estimate("Peak transient vertices",peak.transientVertices);
    estimate("Peak transient indices",peak.transientIndices);
    ImGui::TextWrapped("Peaks cover sampled estimates since start/reset. Delayed snapshots can miss short-lived allocations.");
    ImGui::Separator();
    ImGui::Text("Dynamic VB/IB: %u / %u | Textures: %u",s.numDynamicVertexBuffers,s.numDynamicIndexBuffers,s.numTextures);
    ImGui::Text("Programs: %u | Shaders: %u | Uniforms: %u",s.numPrograms,s.numShaders,s.numUniforms);
    ImGui::TextWrapped("Provisional budgets: 512 MB particle working memory; 1,500 MB total GPU allocations. This snapshot does not certify either budget.");
    ImGui::TextWrapped("Payload excludes alignment, driver copies, shaders, GUI and render targets. Reset can temporarily retain old buffers. Renderer estimates overlap payload; do not add them together.");
    ImGui::End();
}
}
