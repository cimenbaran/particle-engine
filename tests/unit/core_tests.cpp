#include "core/Commands.hpp"
#include "core/CommandQueue.hpp"
#include "core/FixedStep.hpp"
#include "core/Frustum.hpp"
#include "core/ParticleTypes.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#define CHECK(x) do { if (!(x)) { std::cerr << "FAIL line " << __LINE__ << ": " #x "\n"; return EXIT_FAILURE; } } while(false)
int main() {
    const float identityMatrix[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    const auto symmetric=pw::extractFrustum(identityMatrix,true);
    const auto zeroDepth=pw::extractFrustum(identityMatrix,false);
    CHECK(symmetric[0][0]==1 && symmetric[0][3]==1 && symmetric[1][0]==-1);
    CHECK(symmetric[4][2]==1 && symmetric[4][3]==1 && symmetric[5][2]==-1);
    CHECK(zeroDepth[4][2]==1 && zeroDepth[4][3]==0);
    float invalidMatrix[16]{};bool invalidFrustum=false;
    try {pw::extractFrustum(invalidMatrix,true);} catch(const std::exception&) {invalidFrustum=true;}
    CHECK(invalidFrustum);
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
