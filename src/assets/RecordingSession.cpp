#include "RecordingSession.hpp"
#include <stdexcept>
namespace pw {
void RecordingSession::capture(ScenePreset scene,const MaterialLibrary& materials,std::uint32_t capacity,std::uint32_t workgroup) {
    if(active()) throw std::runtime_error("Stop the active recording session first");
    CommandRecording candidate;candidate.scene=std::move(scene);candidate.capacity=capacity;
    candidate.workgroupSize=workgroup;candidate.materialsHash=materials.contentHash;
    validateRecording(candidate,materials);data_=std::move(candidate);mode_=Mode::Capture;
    cursor_=dueBegin_=0;duePending_=false;completed_=0;finished_=false;message_="Recording applied commands from tick zero";
}
void RecordingSession::replay(CommandRecording recording,const MaterialLibrary& materials,std::uint32_t capacity) {
    if(active()) throw std::runtime_error("Stop the active recording session first");
    validateRecording(recording,materials);
    if(recording.capacity!=capacity) throw std::runtime_error("Replay requires the recorded pool capacity; use the CLI for a different capacity");
    data_=std::move(recording);cursor_=dueBegin_=0;duePending_=false;completed_=0;finished_=data_->ticks==0;
    mode_=finished_?Mode::Idle:Mode::Replay;message_=finished_?"Replay complete (zero ticks)":"Replaying recorded commands";
}
void RecordingSession::stop() {
    if(capturing()) message_="Recording stopped; only applied commands are included";
    else if(replaying()) message_="Replay stopped before completion";
    mode_=Mode::Idle;
}
std::span<const Command> RecordingSession::due(std::uint64_t tick) {
    if(!replaying()) return {};
    if(duePending_) throw std::runtime_error("Replay batch already issued");
    if(tick!=completed_) throw std::runtime_error("Replay tick diverged from recording");
    duePending_=true;dueBegin_=cursor_;
    while(cursor_<data_->commands.size() && data_->commands[cursor_].tick==tick) ++cursor_;
    return std::span<const Command>(data_->commands).subspan(dueBegin_,cursor_-dueBegin_);
}
void RecordingSession::observed(std::span<const Command> applied,std::uint64_t completedTicks) {
    if(!active()) return;
    if(completedTicks!=completed_+1) throw std::runtime_error("Recording session skipped a tick");
    if(capturing()) {
        if(data_->commands.size()+applied.size()>maxRecordingCommands || completedTicks>maxRecordingTicks)
            throw std::runtime_error("Recording session exceeded its quota");
        data_->commands.insert(data_->commands.end(),applied.begin(),applied.end());data_->ticks=completedTicks;
        if(data_->commands.size()==maxRecordingCommands || completedTicks==maxRecordingTicks) {
            mode_=Mode::Idle;message_="Recording stopped at its bounded quota";
        }
    } else {
        if(!duePending_) throw std::runtime_error("Replay batch was not requested");
        if(applied.size()!=cursor_-dueBegin_) throw std::runtime_error("Replay command count diverged");
        for(std::size_t i=0;i<applied.size();++i) if(!(applied[i]==data_->commands[dueBegin_+i]))
            throw std::runtime_error("Replay command contents diverged");
        duePending_=false;
        if(completedTicks==data_->ticks) {mode_=Mode::Idle;finished_=true;message_="Replay complete";}
    }
    completed_=completedTicks;
}
}
