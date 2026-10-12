#include "Recordings.hpp"
#include "core/CommandQueue.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace pw {
namespace {
using Json=nlohmann::json;
constexpr std::string_view profile="ballistic_pool_v1";
std::uint64_t integer(const Json& value,std::uint64_t high) {
    if(!value.is_number_integer() || (!value.is_number_unsigned() && value.get<std::int64_t>()<0) || value.get<std::uint64_t>()>high)
        throw std::runtime_error("Recording integer is outside its supported range");
    return value.get<std::uint64_t>();
}
void fields(const Json& json,std::initializer_list<std::string_view> names) {
    if(!json.is_object() || json.size()!=names.size()) throw std::runtime_error("Invalid recording fields");
    for(auto name:names) if(!json.contains(std::string(name))) throw std::runtime_error("Missing recording field: "+std::string(name));
}
std::string hashText(std::uint64_t value) {
    std::ostringstream text;text<<std::hex<<std::setfill('0')<<std::setw(16)<<value;return text.str();
}
const char* typeName(CommandType type) {
    switch(type) {
        case CommandType::SpawnBrush:return "spawn";
        case CommandType::RemoveBrush:return "remove";
        case CommandType::ApplyImpulse:return "impulse";
        default:throw std::runtime_error("Command is not supported by the ballistic recording profile");
    }
}
template<std::size_t N> std::array<float,N> numbers(const Json& json) {
    if(!json.is_array() || json.size()!=N) throw std::runtime_error("Invalid recording vector length");
    for(const auto& value:json) if(!value.is_number()) throw std::runtime_error("Recording vectors require numeric values");
    auto values=json.get<std::array<float,N>>();
    for(auto value:values) if(!std::isfinite(value)) throw std::runtime_error("Recording vectors require finite values");
    return values;
}
}
void validateRecording(const CommandRecording& recording,const MaterialLibrary& materials) {
    parseScene(encodeScene(recording.scene));
    validateSceneCapacity(recording.scene,recording.capacity);sceneMaterials(recording.scene,materials);
    if(recording.materialsHash!=materials.contentHash) throw std::runtime_error("Recording material table hash mismatch");
    if(recording.workgroupSize!=128 && recording.workgroupSize!=256) throw std::runtime_error("Invalid recorded workgroup size");
    if(recording.ticks>maxRecordingTicks || recording.commands.size()>maxRecordingCommands) throw std::runtime_error("Recording duration or command quota exceeded");
    CommandLimits limits;limits.materialCount=std::uint32_t(materials.entries.size());
    std::set<std::uint64_t> sequences;
    std::uint64_t previousTick=0,previousSequence=0,spawnCount=0;
    std::size_t count=0;bool first=true;
    for(const auto& command:recording.commands) {
        typeName(command.type);
        if(auto reason=validate(command,limits,command.tick);!reason.empty()) throw std::runtime_error("Invalid recorded command: "+std::string(reason));
        if(command.tick>=recording.ticks || command.sequence==UINT64_MAX || !sequences.insert(command.sequence).second)
            throw std::runtime_error("Recorded command tick or sequence is invalid");
        if(materials.entries[command.material].gpu.identity[1]!=0) throw std::runtime_error("Recording commands require solid materials");
        if(command.type!=CommandType::SpawnBrush && command.count!=0) throw std::runtime_error("Only spawn commands may have a count");
        if(!first && (command.tick<previousTick || (command.tick==previousTick && command.sequence<=previousSequence)))
            throw std::runtime_error("Recording commands must be ordered by tick and sequence");
        if(first || command.tick!=previousTick) {count=0;spawnCount=0;}
        ++count;if(command.type==CommandType::SpawnBrush) spawnCount+=command.count;
        if(count>CommandQueue::maxPerTick || spawnCount>limits.spawnPerTick) throw std::runtime_error("Recorded tick exceeds command or spawn quota");
        previousTick=command.tick;previousSequence=command.sequence;first=false;
    }
}
std::string encodeRecording(const CommandRecording& recording,const MaterialLibrary& materials) {
    validateRecording(recording,materials);
    Json commands=Json::array();
    for(const auto& command:recording.commands) commands.push_back({{"tick",command.tick},{"sequence",command.sequence},
        {"type",typeName(command.type)},{"minimum",command.minimum},{"maximum",command.maximum},{"parameters",command.parameters},
        {"material",materials.entries[command.material].name},{"count",command.count}});
    Json root={{"schema_version",1},{"simulation_profile",profile},{"tick_hz",60},{"capacity",recording.capacity},
        {"workgroup_size",recording.workgroupSize},{"materials_hash",hashText(recording.materialsHash)},
        {"scene",Json::parse(encodeScene(recording.scene))},{"ticks",recording.ticks},{"commands",commands}};
    auto text=root.dump(2)+"\n";
    if(text.size()>maxRecordingBytes) throw std::runtime_error("Recording exceeds 16 MiB");
    return text;
}
CommandRecording parseRecording(std::string_view text,const MaterialLibrary& materials) {
    if(text.size()>maxRecordingBytes) throw std::runtime_error("Recording exceeds 16 MiB");
    const auto root=Json::parse(text);
    fields(root,{"schema_version","simulation_profile","tick_hz","capacity","workgroup_size","materials_hash","scene","ticks","commands"});
    if(integer(root.at("schema_version"),1)!=1 || root.at("simulation_profile")!=profile || integer(root.at("tick_hz"),60)!=60)
        throw std::runtime_error("Unsupported recording version, simulation profile or tick rate");
    if(root.at("materials_hash")!=hashText(materials.contentHash)) throw std::runtime_error("Recording material table hash mismatch");
    CommandRecording recording;recording.materialsHash=materials.contentHash;
    recording.capacity=std::uint32_t(integer(root.at("capacity"),1048576));
    recording.workgroupSize=std::uint32_t(integer(root.at("workgroup_size"),256));
    recording.ticks=integer(root.at("ticks"),maxRecordingTicks);
    recording.scene=parseScene(root.at("scene").dump());
    const auto& entries=root.at("commands");
    if(!entries.is_array() || entries.size()>maxRecordingCommands) throw std::runtime_error("Recording command quota exceeded");
    for(const auto& entry:entries) {
        fields(entry,{"tick","sequence","type","minimum","maximum","parameters","material","count"});
        Command command;command.tick=integer(entry.at("tick"),maxRecordingTicks);command.sequence=integer(entry.at("sequence"),UINT64_MAX-1);
        const auto type=entry.at("type").get<std::string>();
        if(type=="spawn") command.type=CommandType::SpawnBrush;
        else if(type=="remove") command.type=CommandType::RemoveBrush;
        else if(type=="impulse") command.type=CommandType::ApplyImpulse;
        else throw std::runtime_error("Unsupported recorded command type");
        command.minimum=numbers<3>(entry.at("minimum"));command.maximum=numbers<3>(entry.at("maximum"));
        command.parameters=numbers<4>(entry.at("parameters"));command.count=std::uint32_t(integer(entry.at("count"),4096));
        const auto name=entry.at("material").get<std::string>();
        const auto found=std::find_if(materials.entries.begin(),materials.entries.end(),[&](const auto& material){return material.name==name;});
        if(found==materials.entries.end()) throw std::runtime_error("Unknown recorded material: "+name);
        command.material=std::uint32_t(found-materials.entries.begin());recording.commands.push_back(command);
    }
    validateRecording(recording,materials);return recording;
}
CommandRecording loadRecording(const std::filesystem::path& path,const MaterialLibrary& materials) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input || input.tellg()<0 || input.tellg()>std::streamoff(maxRecordingBytes)) throw std::runtime_error("Cannot read recording or file exceeds 16 MiB");
    std::string text(std::size_t(input.tellg()),'\0');input.seekg(0);input.read(text.data(),std::streamsize(text.size()));
    if(!input) throw std::runtime_error("Cannot read complete recording");return parseRecording(text,materials);
}
void saveRecording(const std::filesystem::path& path,const CommandRecording& recording,const MaterialLibrary& materials) {
    const auto text=encodeRecording(recording,materials);
    if(path.empty() || path.filename().empty()) throw std::runtime_error("Recording destination must be a file");
    if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    static std::atomic<std::uint64_t> saveSequence{};
    auto temporaryDirectory=path;
    temporaryDirectory+=std::string(".tmp.")+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"."+std::to_string(saveSequence.fetch_add(1));
    // Directory creation is exclusive, unlike an exists-then-open file check.
    // A rare name collision fails without touching another writer's payload.
    if(!std::filesystem::create_directory(temporaryDirectory)) throw std::runtime_error("Recording temporary path already exists");
    const auto temporary=temporaryDirectory/"payload.json";
    try {
        std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
        output.write(text.data(),std::streamsize(text.size()));output.close();
        if(!output) throw std::runtime_error("Cannot write complete recording");
#if defined(_WIN32)
        if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot replace recording file");
#else
        std::filesystem::rename(temporary,path);
#endif
    } catch(...) {std::error_code ignored;std::filesystem::remove_all(temporaryDirectory,ignored);throw;}
    std::error_code ignored;std::filesystem::remove(temporaryDirectory,ignored);
}
}
