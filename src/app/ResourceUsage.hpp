#pragma once
#include <bgfx/bgfx.h>
#include "core/ReportedMemory.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
namespace pw {
// Requested payload sizes, not physical/resident driver allocations.
struct ResourceUsage {
    std::uint64_t state{},instances{},lists{},counters{},indirect{},materials{},geometry{},terrainGeometry{};
    std::uint64_t diagnosticTextures{},diagnosticHost{},auditBuffers{},seedUpload{},granular{},contactNormals{};
    ResourceUsage& operator+=(const ResourceUsage& other);
    std::uint64_t particleBuffers() const {return state+instances+lists+counters+indirect+granular+contactNormals;}
    std::uint64_t persistentPayload() const {return particleBuffers()+materials+geometry+terrainGeometry+diagnosticTextures+auditBuffers;}
};
ResourceUsage resourceUsage(std::uint32_t capacity,std::uint32_t materialCount,bool indirectAllocated,bool audit=false);
std::uint64_t contactNormalPayloadBytes(std::uint32_t capacity);
std::uint64_t settlingPayloadBytes(std::uint32_t capacity);
std::uint64_t granularPayloadBytes(std::uint32_t capacity);
nlohmann::json resourceReport(const ResourceUsage& usage,const bgfx::Stats& stats);
ReportedMemory reportedMemory(const bgfx::Stats& stats);
nlohmann::json resourcePeakReport(const ReportedMemoryPeaks& peaks,const char* scope);
void drawResourceUsage(const ResourceUsage& usage,const bgfx::Stats& stats,ReportedMemoryPeaks& peaks);
}
