#include "core/Commands.hpp"
#include "core/ReportedMemory.hpp"
#include "core/ComputeDispatch.hpp"
#include "core/CommandQueue.hpp"
#include "core/FixedStep.hpp"
#include "core/FlyCamera.hpp"
#include "core/PointerBrush.hpp"
#include "core/Frustum.hpp"
#include "core/RunOptions.hpp"
#include "core/TimingSummary.hpp"
#include <string>
#include <vector>
#include "core/ParticleTypes.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#define CHECK(x) do { if (!(x)) { std::cerr << "FAIL line " << __LINE__ << ": " #x "\n"; return EXIT_FAILURE; } } while(false)
int main() {
    pw::ReportedMemoryPeaks peaks;
    CHECK(peaks.observations()==0 && peaks.values().gpu<0);
    peaks.observe({-1,400,800,0,12});
    peaks.observe({-1,200,1200,100,4});
    peaks.observe({-INT64_MAX,-1,-1,-1,-1});
    CHECK(peaks.observations()==3 && peaks.values().gpu<0);
    CHECK(peaks.values().textures==400 && peaks.values().renderTargets==1200);
    CHECK(peaks.values().transientVertices==100 && peaks.values().transientIndices==12);
    peaks.reset();CHECK(peaks.observations()==0 && peaks.values().textures<0);
    peaks.observe({0,0,0,0,0});CHECK(peaks.values().gpu==0 && peaks.values().textures==0);
    peaks.observe({INT64_MAX,-1,-1,-1,-1});peaks.observe({1,-1,-1,-1,-1});
    CHECK(peaks.values().gpu==INT64_MAX);

    const auto centerRay=pw::pointerRay({0,10,10},{0,0,0},.5f,.5f,16.0f/9);
    CHECK(centerRay);const auto centerBox=pw::planeBrush(*centerRay,0,1);CHECK(centerBox);
    CHECK(std::abs(centerBox->minimum[0]+1)<1e-5f && std::abs(centerBox->maximum[2]-1)<1e-5f);
    CHECK(centerBox->minimum[1]==0 && centerBox->maximum[1]==1);
    const auto sideRay=pw::pointerRay({0,10,10},{0,0,0},.6f,.5f,16.0f/9);CHECK(sideRay && sideRay->direction[0]>0);
    CHECK(!pw::pointerRay({0,10,10},{0,10,10},.5f,.5f,1));
    CHECK(!pw::pointerRay({0,10,10},{0,0,0},-1,.5f,1));
    CHECK(!pw::planeBrush({{0,10,10},{0,0,-1}},0,1));
    CHECK(!pw::planeBrush({{0,10,10},{0,1,0}},0,1));
    CHECK(!pw::planeBrush({{20,10,0},{0,-1,0}},0,1));
    const auto edgeBox=pw::planeBrush({{7.9f,10,7.9f},{0,-1,0}},0,1);CHECK(edgeBox);
    CHECK(edgeBox->maximum[0]==7.95f && edgeBox->maximum[2]==7.95f);

    pw::FlyCamera closeCamera;closeCamera.fromOrbit(0,8,5,1);
    CHECK(std::abs(closeCamera.forward()[1]+1/std::sqrt(5.0f))<1e-6f && std::abs(closeCamera.forward()[2]+2/std::sqrt(5.0f))<1e-6f);
    pw::FlyCamera camera;camera.fromOrbit(0,10,3);
    const auto forward=camera.forward();CHECK(std::abs(forward[0])<1e-6f && forward[1]==0 && forward[2]==-1);
    camera.move(1,0,0,.1f,10);CHECK(std::abs(camera.position()[2]-9)<1e-6f);
    camera.fromOrbit(0,10,3);camera.move(1,1,0,.1f,10);
    const auto diagonal=camera.position();CHECK(std::abs(std::hypot(diagonal[0],diagonal[2]-10)-1)<1e-5f);
    camera.fromOrbit(0,10,3);camera.move(1,0,0,10,10);CHECK(camera.position()[2]==9); // stall clamp
    camera.look(100,0);CHECK(camera.forward()[0]>0);
    camera.look(0,-1e6f);CHECK(camera.forward()[1]<1 && camera.forward()[1]>.99f); // no pole singularity
    const auto finiteDirection=camera.forward();camera.look(std::numeric_limits<float>::infinity(),0);
    CHECK(camera.forward()==finiteDirection);
    for(unsigned i=0;i<1000;++i) camera.move(0,0,-1,.1f,100);
    CHECK(camera.position()[1]==.1f);

    CHECK(pw::dispatchGroups(0)==0);
    CHECK(pw::dispatchGroups(1)==1);
    CHECK(pw::dispatchGroups(pw::workgroupSize-1)==1);
    CHECK(pw::dispatchGroups(pw::workgroupSize)==1);
    CHECK(pw::dispatchGroups(pw::workgroupSize+1)==2);
    CHECK(pw::dispatchGroups(1048576)==1048576/pw::workgroupSize);
    CHECK(pw::dispatchGroups(UINT32_MAX)==uint64_t(UINT32_MAX)/pw::workgroupSize+1);

    const float identityMatrix[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    const auto screenPoint=pw::projectBrushPoint({-.5f,.5f,0},identityMatrix,1280,720);
    CHECK(screenPoint && (*screenPoint)[0]==320 && (*screenPoint)[1]==180);
    const auto logicalRay=pw::pointerRay({0,10,10},{0,0,0},640.0f/1280,360.0f/720,16.0f/9);
    const auto retinaRay=pw::pointerRay({0,10,10},{0,0,0},1280.0f/2560,720.0f/1440,16.0f/9);
    CHECK(logicalRay && retinaRay && logicalRay->direction==retinaRay->direction);
    const auto symmetric=pw::extractFrustum(identityMatrix,true);
    const auto zeroDepth=pw::extractFrustum(identityMatrix,false);
    CHECK(symmetric[0][0]==1 && symmetric[0][3]==1 && symmetric[1][0]==-1);
    CHECK(symmetric[4][2]==1 && symmetric[4][3]==1 && symmetric[5][2]==-1);
    CHECK(zeroDepth[4][2]==1 && zeroDepth[4][3]==0);
    float invalidMatrix[16]{};bool invalidFrustum=false;
    try {pw::extractFrustum(invalidMatrix,true);} catch(const std::exception&) {invalidFrustum=true;}
    CHECK(invalidFrustum);
    std::vector<double> timings;
    for(int i=100;i>=1;--i) timings.push_back(i);
    const auto timing=pw::summarize(timings);
    CHECK(timing.minimum==1 && timing.median==50 && timing.p95==95 && timing.maximum==100 && timing.mean==50.5);
    bool badTiming=false;try {pw::summarize({});} catch(const std::exception&) {badTiming=true;}CHECK(badTiming);
    auto parse=[](std::vector<std::string> args) {
        std::vector<char*> raw;for(auto& arg:args) raw.push_back(arg.data());return pw::parseOptions(int(raw.size()),raw.data());
    };
    CHECK(parse({"app","--scene","scene.json","--frames","2"}).scene=="scene.json");
    CHECK(parse({"app"}).capacity==16384);
    CHECK(parse({"app","--granular-check","--report","check.json","--software-gpu"}).granularCheck);
    CHECK(parse({"app","--granular"}).granular && parse({"app","--granular"}).capacity==512 && parse({"app","--granular"}).cosmeticCapacity==0);
    CHECK(parse({"app","--granular","--capacity","129","--frames","8"}).capacity==129);
    CHECK(parse({"app","--granular","--soak","600","--report","x","--software-gpu"}).softwareGpu);
    CHECK(parse({"app","--granular","--soak","600","--report","x"}).capacity==512);
    CHECK(parse({"app","--granular","--scene","sphere.json","--soak","600","--report","x","--software-gpu"}).scene=="sphere.json");
    CHECK(parse({"app","--gpu-test","--software-gpu"}).softwareGpu);
    CHECK(parse({"app","--replay","x","--software-gpu"}).softwareGpu);
    CHECK(parse({"app","--capabilities","x","--software-gpu"}).softwareGpu);
    CHECK(parse({"app","--cosmetic-capacity","0"}).cosmeticCapacity==0);
    CHECK(parse({"app","--benchmark","x"}).cosmeticCapacity==0);
    CHECK(parse({"app","--cosmetic-capacity","150000","--benchmark","x"}).cosmeticCapacity==150000);
    CHECK(parse({"app","--benchmark","x","--cosmetic-capacity","0"}).cosmeticCapacity==0);
    CHECK(parse({"app","--cosmetic-capacity","150000","--capacity","1","--frames","2"}).cosmeticCapacity==150000);
    CHECK(parse({"app","--replay","capture.json","--report","result.json","--no-indirect"}).replay=="capture.json");
    CHECK(parse({"app","--capabilities","device.json"}).capabilities=="device.json");
    CHECK(parse({"app","--soak","600","--report","soak.json"}).soakSeconds==600);
    const auto run=parse({"app","--capacity","1000000","--benchmark","report","--warmup","0","--no-indirect"});
    CHECK(run.capacity==1000000 && run.frames==600 && run.warmup==0 && !run.indirect);
    for(const auto& args:std::vector<std::vector<std::string>>{
        {"app","--software-gpu"},{"app","--software-gpu","--frames","2"},
        {"app","--software-gpu","--benchmark","x"},
        {"app","--software-gpu","--soak","1","--report","x"},
        {"app","--cosmetic-capacity","150001"},{"app","--cosmetic-capacity","-1"},
        {"app","--cosmetic-capacity","1","--replay","x"},
        {"app","--cosmetic-capacity","1","--gpu-test"},{"app","--cosmetic-capacity","1","--capabilities","x"},
        {"app","--cosmetic-capacity","1","--soak","1","--report","x"},
        {"app","--replay",""},{"app","--replay","x","--capacity","129"},
        {"app","--replay","x","--frames","1"},{"app","--replay","x","--warmup","0"},
        {"app","--replay","x","--scene","y"},{"app","--replay","x","--gpu-test"},
        {"app","--replay","x","--benchmark","y"},{"app","--replay","x","--capabilities","y"},
        {"app","--replay","x","--soak","1","--report","y"},
        {"app","--granular-check"},{"app","--granular-check","--report","x","--frames","2"},
        {"app","--granular-check","--report","x","--capacity","64"},{"app","--granular-check","--report","x","--granular"},
        {"app","--granular-check","--report","x","--gpu-test"},{"app","--granular","--gpu-test"},{"app","--granular","--replay","x"},
        {"app","--granular","--benchmark","x"},{"app","--granular","--soak","10"},
        {"app","--granular","--software-gpu"},{"app","--granular","--soak","10","--report","x","--frames","1"},
        {"app","--scene",""},{"app","--scene","x","--benchmark","y"},
        {"app","--scene","x","--soak","10","--report","x"},{"app","--scene","x","--capabilities","x"},
        {"app","--scene","x","--gpu-test"},{"app","--capacity","-1"},{"app","--capacity","1048577"},{"app","--frames","2oops"},
        {"app","--frames","4294967296"},{"app","--frames","0"},{"app","--capacity"},
        {"app","--warmup","1"},{"app","--gpu-test","--benchmark","x"},
        {"app","--capabilities",""},{"app","--capabilities","x","--gpu-test"},
        {"app","--capabilities","x","--frames","2"},{"app","--capabilities","x","--soak","1","--report","y"},
        {"app","--soak","0","--report","x"},{"app","--soak","3601","--report","x"},
        {"app","--soak","1"},{"app","--report","x"},{"app","--soak","1","--report","x","--frames","2"}}) {
        bool failed=false;try {parse(args);} catch(const std::exception&) {failed=true;}CHECK(failed);
    }
    pw::FixedStep clock;
    CHECK(clock.advance(1.0/120, false, false).count == 0);
    CHECK(clock.advance(1.0/120, false, false).count == 1);
    const auto lag = clock.advance(1.0, false, false);
    CHECK(lag.count == 4 && lag.droppedSeconds > 0.9);
    CHECK(clock.advance(0, false, false).count == 0);
    CHECK(clock.advance(9, true, false).count == 0);
    CHECK(clock.advance(0, true, true).count == 1);
    CHECK(clock.advance(std::numeric_limits<double>::infinity(), false, false).count == 0);
    CHECK(clock.advance(-1, false, false).count == 0);
    CHECK(clock.advance(1e100, false, false).count == 4);
    pw::Command c; c.count = 128;
    CHECK(pw::validate(c, {}, 0).empty());
    c.material = 7; CHECK(!pw::validate(c, {}, 0).empty()); c.material = 0;
    c.count = 4097; CHECK(!pw::validate(c, {}, 0).empty()); c.count = 1;
    c.minimum[0] = 1; CHECK(!pw::validate(c, {}, 0).empty()); c.minimum[0] = 0;
    c.parameters[1] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!pw::validate(c, {}, 0).empty()); c.parameters[1] = 0;
    CHECK(!pw::validate(c, {}, 1).empty());
    c.type = static_cast<pw::CommandType>(99); CHECK(!pw::validate(c, {}, 0).empty());
    CHECK((pw::ParticleId{3, 1} != pw::ParticleId{3, 2}));
    c={}; c.type=pw::CommandType::SpawnBrush; c.parameters[3]=-1;
    CHECK(!pw::validate(c,{},0).empty()); c.parameters[3]=3601;
    CHECK(!pw::validate(c,{},0).empty());
    pw::CommandQueue queue;
    c={}; c.type=pw::CommandType::ApplyImpulse; c.tick=1;
    CHECK(queue.enqueue(c,{}).empty());
    c.tick=0; CHECK(queue.enqueue(c,{}).empty());
    auto commands=queue.consume();
    CHECK(commands.size()==1 && commands[0].tick==0 && commands[0].sequence==1);
    CHECK(!queue.enqueue(c,{}).empty());
    commands=queue.consume();
    CHECK(commands.size()==1 && commands[0].tick==1 && commands[0].sequence==0);
    CHECK(queue.consume().empty());
    queue.reset(); c={};
    c.count=4096;
    CHECK(queue.enqueue(c,{}).empty()); CHECK(queue.enqueue(c,{}).empty());
    CHECK(!queue.enqueue(c,{}).empty()); CHECK(queue.pending()==2);
    queue.reset(); c={}; c.type=pw::CommandType::ApplyImpulse;
    for(unsigned i=0;i<32;++i) CHECK(queue.enqueue(c,{}).empty());
    CHECK(!queue.enqueue(c,{}).empty());
    commands=queue.consume(); CHECK(commands.size()==32);
    for(unsigned i=0;i<32;++i) CHECK(commands[i].sequence==i);
    queue.reset(); c.tick=121; CHECK(!queue.enqueue(c,{}).empty());
    c.tick=120; CHECK(queue.enqueue(c,{}).empty());
    queue.reset(); c.tick=0; c.parameters[0]=101; CHECK(!queue.enqueue(c,{}).empty());
    c.parameters[0]=0;
    for(unsigned tick=0;tick<8;++tick) {
        c.tick=tick;
        for(unsigned i=0;i<32;++i) CHECK(queue.enqueue(c,{}).empty());
    }
    c.tick=8; CHECK(!queue.enqueue(c,{}).empty()); CHECK(queue.pending()==256);
    queue.reset(); CHECK(queue.pending()==0 && queue.rejected()==0 && queue.nextTick()==0);
    std::cout << "Core invariants passed\n";
}
