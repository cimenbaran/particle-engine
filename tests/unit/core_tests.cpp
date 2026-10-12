#include "core/Commands.hpp"
#include "core/FixedStep.hpp"
#include "core/ParticleTypes.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#define CHECK(x) do { if (!(x)) { std::cerr << "FAIL line " << __LINE__ << ": " #x "\n"; return EXIT_FAILURE; } } while(false)
int main() {
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
    std::cout << "Core invariants passed\n";
}
