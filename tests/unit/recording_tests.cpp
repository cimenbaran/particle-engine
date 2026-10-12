#include "assets/RecordingSession.hpp"
#include "core/CommandQueue.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do {if(!(x)) throw std::runtime_error("Check failed: " #x);} while(false)
int main(int argc,char** argv) {
 try {
    CHECK(argc==3);const std::filesystem::path assets=argv[1],directory=argv[2];
    const auto materials=pw::loadMaterials(assets/"materials/default.json");
    pw::CommandRecording recording;recording.scene=pw::loadScene(assets/"scenes/empty.json");
    recording.capacity=129;recording.ticks=121;recording.materialsHash=materials.contentHash;recording.workgroupSize=256;
    pw::Command command;command.type=pw::CommandType::SpawnBrush;command.count=129;command.sequence=1;
    command.minimum={-1,2,-1};command.maximum={1,3,1};recording.commands.push_back(command);
    command.type=pw::CommandType::ApplyImpulse;command.count=0;command.sequence=2;command.parameters[1]=2;recording.commands.push_back(command);
    command.tick=120;command.sequence=0;command.type=pw::CommandType::RemoveBrush;command.parameters={};recording.commands.push_back(command);
    pw::RecordingSession session;
    session.replay(recording,materials,129);
    pw::CommandQueue replayQueue;
    for(std::uint64_t tick=0;tick<recording.ticks;++tick) {
        for(const auto& due:session.due(tick)) CHECK(replayQueue.enqueueRecorded(due,{}).empty());
        session.observed(replayQueue.consume(),tick+1);
    }
    CHECK(session.finished() && !session.active() && session.completedTicks()==121);
    bool mismatch=false;
    try {session.replay(recording,materials,128);}catch(const std::exception&){mismatch=true;}
    CHECK(mismatch && session.finished() && session.completedTicks()==121);
    session.capture(recording.scene,materials,129,128);
    session.observed(std::span<const pw::Command>(recording.commands).first(2),1);
    session.observed({},2);session.stop();
    CHECK(session.recording().ticks==2 && session.recording().commands.size()==2);
    session.replay(session.recording(),materials,129);
    mismatch=false;
    try {session.observed({},1);}catch(const std::exception&){mismatch=true;}
    CHECK(mismatch && session.completedTicks()==0);
    mismatch=false;
    try {session.due(1);}catch(const std::exception&){mismatch=true;}CHECK(mismatch);
    session.due(0);mismatch=false;
    try {session.due(0);}catch(const std::exception&){mismatch=true;}CHECK(mismatch);
    mismatch=false;
    try {session.observed({},1);}catch(const std::exception&){mismatch=true;}
    CHECK(mismatch && session.completedTicks()==0);session.stop();
    session.capture(recording.scene,materials,129,128);
    for(std::uint64_t tick=0;tick<pw::maxRecordingTicks;++tick) session.observed({},tick+1);
    CHECK(!session.active() && session.recording().ticks==pw::maxRecordingTicks);
    auto zero=recording;zero.ticks=0;zero.commands.clear();session.replay(zero,materials,129);
    CHECK(session.finished() && !session.active() && session.completedTicks()==0);
    const auto encoded=pw::encodeRecording(recording,materials);
    const auto parsed=pw::parseRecording(encoded,materials);
    CHECK(pw::encodeRecording(parsed,materials)==encoded && parsed.workgroupSize==256);
    // A future command accepted before earlier-tick work retains its lower ID.
    pw::CommandQueue queue;std::size_t cursor=0,applied=0;
    for(std::uint64_t tick=0;tick<parsed.ticks;++tick) {
        while(cursor<parsed.commands.size() && parsed.commands[cursor].tick==tick)
            CHECK(queue.enqueueRecorded(parsed.commands[cursor++],{}).empty());
        const auto batch=queue.consume();
        for(const auto& actual:batch) {
            const auto& expected=parsed.commands[applied++];
            CHECK(actual.tick==expected.tick && actual.sequence==expected.sequence && actual.type==expected.type);
            CHECK(actual.minimum==expected.minimum && actual.maximum==expected.maximum && actual.parameters==expected.parameters);
            CHECK(actual.count==expected.count && actual.material==expected.material);
        }
    }
    CHECK(applied==3 && queue.pending()==0 && queue.nextTick()==121);
    command.tick=121;command.sequence=0;CHECK(queue.enqueue(command,{}).empty());CHECK(queue.consume().front().sequence==3);
    // Explicit replay IDs near exhaustion do not prevent older recorded IDs.
    queue.reset();command.tick=0;command.sequence=UINT64_MAX-1;
    CHECK(queue.enqueueRecorded(command,{}).empty());queue.consume();
    command.tick=1;command.sequence=0;CHECK(queue.enqueueRecorded(command,{}).empty());queue.consume();
    command.tick=2;CHECK(!queue.enqueue(command,{}).empty());
    const auto source=nlohmann::json::parse(encoded);
    auto rejects=[&](auto change) {
        auto candidate=source;change(candidate);bool rejected=false;
        try {pw::parseRecording(candidate.dump(),materials);}catch(const std::exception&){rejected=true;}
        if(!rejected) throw std::runtime_error("Accepted invalid recording: "+candidate.dump());
    };
    rejects([](auto& j){j["schema_version"]=2;});rejects([](auto& j){j["tick_hz"]=30;});
    rejects([](auto& j){j["simulation_profile"]="future_fluid";});rejects([](auto& j){j["capacity"]=0;});
    rejects([](auto& j){j["materials_hash"]="0000000000000000";});rejects([](auto& j){j["workgroup_size"]=64;});
    rejects([](auto& j){j["ticks"]=216001;});rejects([](auto& j){j["commands"][0]["tick"]=121;});
    rejects([](auto& j){j["commands"][0]["tick"]=-1;});rejects([](auto& j){j["commands"][0]["count"]=4097;});
    rejects([](auto& j){j["commands"][0]["sequence"]=UINT64_MAX;});
    rejects([](auto& j){j["commands"][1]["sequence"]=1;});rejects([](auto& j){std::swap(j["commands"][0],j["commands"][1]);});
    rejects([](auto& j){j["commands"][0]["minimum"][0]=true;});
    rejects([](auto& j){j["commands"][0]["maximum"][0]=2048;});
    rejects([](auto& j){j["commands"][0]["parameters"]={0,0,0};});
    rejects([](auto& j){j["commands"][0]["parameters"][0]=101;});
    rejects([](auto& j){j["commands"][0]["material"]="Water";});
    rejects([](auto& j){j["commands"][0]["material"]="Unknown";});
    rejects([](auto& j){j["commands"][0]["type"]="heat";});
    rejects([](auto& j){j["commands"][1]["count"]=1;});
    rejects([](auto& j){j["commands"][0]["unused"]=0;});
    rejects([](auto& j){j["commands"]=nlohmann::json::array();for(int i=0;i<33;++i) j["commands"].push_back({{"tick",0},{"sequence",i},{"type","remove"},{"minimum",{0,0,0}},{"maximum",{1,1,1}},{"parameters",{0,0,0,0}},{"material","Stone"},{"count",0}});});
    rejects([](auto& j){auto spawn=j["commands"][0];spawn["count"]=4096;j["commands"]=nlohmann::json::array();for(int i=0;i<3;++i){spawn["sequence"]=i;j["commands"].push_back(spawn);}});
    auto maximum=recording;maximum.commands.clear();maximum.ticks=512;
    command={};command.type=pw::CommandType::RemoveBrush;command.maximum={1,1,1};
    for(std::size_t i=0;i<pw::maxRecordingCommands;++i) {
        command.tick=i/32;command.sequence=i;maximum.commands.push_back(command);
    }
    session.capture(recording.scene,materials,129,128);
    for(std::uint64_t tick=0;tick<512;++tick)
        session.observed(std::span<const pw::Command>(maximum.commands).subspan(tick*32,32),tick+1);
    CHECK(!session.active() && session.recording().commands.size()==pw::maxRecordingCommands);
    pw::validateRecording(session.recording(),materials);
    const auto maximumText=pw::encodeRecording(maximum,materials);
    CHECK(pw::parseRecording(maximumText,materials).commands.size()==pw::maxRecordingCommands);
    maximum.commands.push_back(command);bool totalRejected=false;
    try{pw::validateRecording(maximum,materials);}catch(const std::exception&){totalRejected=true;}CHECK(totalRejected);
    bool bytesRejected=false;
    try{pw::parseRecording(std::string(pw::maxRecordingBytes+1,' '),materials);}catch(const std::exception&){bytesRejected=true;}CHECK(bytesRejected);
    auto duration=recording;duration.commands.clear();duration.ticks=pw::maxRecordingTicks;
    pw::validateRecording(duration,materials);
    // Serialize before touching an existing destination, then atomically replace.
    std::filesystem::create_directories(directory);const auto file=directory/"roundtrip.json";
    pw::saveRecording(file,recording,materials);CHECK(pw::encodeRecording(pw::loadRecording(file,materials),materials)==encoded);
    auto invalid=recording;++invalid.materialsHash;bool failed=false;
    try{pw::saveRecording(file,invalid,materials);}catch(const std::exception&){failed=true;}CHECK(failed);
    CHECK(pw::encodeRecording(pw::loadRecording(file,materials),materials)==encoded);
    auto empty=recording;empty.commands.clear();empty.ticks=0;
    pw::saveRecording(file,empty,materials);CHECK(pw::loadRecording(file,materials).ticks==0);
    auto alternate=recording;++alternate.ticks;
    auto firstWriter=std::async(std::launch::async,[&]{pw::saveRecording(file,recording,materials);});
    auto secondWriter=std::async(std::launch::async,[&]{pw::saveRecording(file,alternate,materials);});
    firstWriter.get();secondWriter.get();
    const auto concurrent=pw::loadRecording(file,materials);
    CHECK((concurrent.ticks==recording.ticks || concurrent.ticks==alternate.ticks) && concurrent.commands.size()==3);
    const auto blocked=directory/"destination-directory";std::filesystem::create_directories(blocked);failed=false;
    try{pw::saveRecording(blocked,recording,materials);}catch(const std::exception&){failed=true;}CHECK(failed && std::filesystem::is_directory(blocked));
    for(const auto& entry:std::filesystem::directory_iterator(directory)) CHECK(entry.path().filename().string().find(".tmp.")==std::string::npos);
    std::cout<<"Recording schema, quotas, ordering, preserved replay IDs and atomic replacement passed\n";
 } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
