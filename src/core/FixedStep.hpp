#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace pw {
struct TickBatch { unsigned count{}; double droppedSeconds{}; };
class FixedStep {
public:
    static constexpr double tickSeconds = 1.0 / 60.0;
    static constexpr unsigned maxCatchUp = 4;
    TickBatch advance(double elapsed, bool paused, bool step) {
        if (paused) { accumulator_ = 0; return {step ? 1u : 0u, 0}; }
        if (!std::isfinite(elapsed) || elapsed < 0) elapsed = 0;
        // Bound before accumulation/conversion, even after a very long suspend.
        const double accepted = std::min(elapsed, tickSeconds * maxCatchUp);
        const double dropped = elapsed - accepted;
        accumulator_ += accepted;
        const auto count = std::min(maxCatchUp,
            static_cast<unsigned>((accumulator_ + 1e-12) / tickSeconds));
        accumulator_ = std::max(0.0, accumulator_ - count * tickSeconds);
        return {count, dropped};
    }
    void reset() { accumulator_ = 0; }
private:
    double accumulator_{};
};
}
