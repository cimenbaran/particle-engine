#pragma once
#include "Scenes.hpp"
#include "core/Commands.hpp"
namespace pw {
inline constexpr std::uint64_t maxRecordingTicks=216000;
inline constexpr std::size_t maxRecordingCommands=16384,maxRecordingBytes=16*1024*1024;
struct CommandRecording {
    ScenePreset scene;
    std::uint32_t capacity=16384,workgroupSize=128;
    std::uint64_t materialsHash{},ticks{};
    // Applied commands in tick order and original sequence order within a tick.
    // Rejected CPU submissions and unconsumed future commands are not recorded.
    std::vector<Command> commands;
};
void validateRecording(const CommandRecording& recording,const MaterialLibrary& materials);
std::string encodeRecording(const CommandRecording& recording,const MaterialLibrary& materials);
CommandRecording parseRecording(std::string_view text,const MaterialLibrary& materials);
CommandRecording loadRecording(const std::filesystem::path& path,const MaterialLibrary& materials);
// Validate first, write a file in an exclusively created sibling temporary
// directory, then replace the destination on the same filesystem.
// A failed write preserves the old destination; this does not snapshot the GPU.
void saveRecording(const std::filesystem::path& path,const CommandRecording& recording,const MaterialLibrary& materials);
}
