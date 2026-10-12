#pragma once
#include "Recordings.hpp"
namespace pw {
class RecordingSession {
public:
    void capture(ScenePreset scene,const MaterialLibrary& materials,std::uint32_t capacity,std::uint32_t workgroup);
    void replay(CommandRecording recording,const MaterialLibrary& materials,std::uint32_t capacity);
    void stop();
    bool capturing() const {return mode_==Mode::Capture;}
    bool replaying() const {return mode_==Mode::Replay;}
    bool active() const {return mode_!=Mode::Idle;}
    bool hasRecording() const {return data_.has_value();}
    bool finished() const {return finished_;}
    std::uint64_t completedTicks() const {return completed_;}
    const CommandRecording& recording() const {return data_.value();}
    const std::string& message() const {return message_;}
    std::span<const Command> due(std::uint64_t tick);
    void observed(std::span<const Command> applied,std::uint64_t completedTicks);
private:
    enum class Mode {Idle,Capture,Replay};
    Mode mode_=Mode::Idle;
    std::optional<CommandRecording> data_;
    std::size_t cursor_=0,dueBegin_=0;
    std::uint64_t completed_=0;
    bool finished_=false,duePending_=false;
    std::string message_;
};
}
