#pragma once
#include "Commands.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace pw {
// Bounded, tick-addressed commands. Sequence numbers never repeat within an epoch.
class CommandQueue {
public:
    static constexpr std::size_t maxPerTick=32, maxPending=256;
    static constexpr std::uint64_t maxFutureTicks=120;
    std::string_view enqueue(Command command, const CommandLimits& limits) {return insert(command,limits,{});}
    // Replay validates unique sequences across the complete file before feeding
    // one tick at a time. Preserve IDs because GPU spawn randomness uses them.
    std::string_view enqueueRecorded(Command command,const CommandLimits& limits) {return insert(command,limits,command.sequence);}
private:
    std::string_view insert(Command command,const CommandLimits& limits,std::optional<std::uint64_t> recordedSequence) {
        auto reject=[&](std::string_view reason) { ++rejected_; return reason; };
        if(auto reason=validate(command,limits,nextTick_);!reason.empty()) return reject(reason);
        if(command.tick-nextTick_>maxFutureTicks) return reject("command is too far in the future");
        if(pending_.size()>=maxPending) return reject("command queue is full");
        std::size_t count=0;
        std::uint64_t spawns=command.type==CommandType::SpawnBrush?command.count:0;
        for(const auto& queued:pending_) if(queued.tick==command.tick) {
            ++count;
            if(queued.type==CommandType::SpawnBrush) spawns+=queued.count;
        }
        if(count>=maxPerTick) return reject("tick command quota exceeded");
        if(spawns>limits.spawnPerTick) return reject("tick spawn quota exceeded");
        if((!recordedSequence && sequence_==std::numeric_limits<std::uint64_t>::max()) || recordedSequence==std::numeric_limits<std::uint64_t>::max()) return reject("command sequence exhausted");
        command.sequence=recordedSequence.value_or(sequence_);
        sequence_=std::max(sequence_,command.sequence+1);
        pending_.push_back(command);
        return {};
    }
public:
    // Consume exactly one tick. Future entries stay queued; reset starts a new epoch.
    std::vector<Command> consume() {
        std::vector<Command> batch;
        for(const auto& command:pending_) if(command.tick==nextTick_) batch.push_back(command);
        std::erase_if(pending_,[&](const Command& command) { return command.tick==nextTick_; });
        ++nextTick_;
        return batch;
    }
    void reset() { pending_.clear(); nextTick_=sequence_=rejected_=0; }
    std::uint64_t nextTick() const {return nextTick_;}
    std::uint64_t rejected() const {return rejected_;}
    std::size_t pending() const {return pending_.size();}
private:
    std::vector<Command> pending_;
    std::uint64_t nextTick_{},sequence_{},rejected_{};
};
}
