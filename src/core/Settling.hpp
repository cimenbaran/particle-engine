#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace pw {
struct SettlingProfile {
    float enterSpeed=0.05f,exitSpeed=0.1f;
    std::uint32_t acquireTicks=30,releaseTicks=3;
};
// Sidecar history for one physical slot. Never stores or changes particle flags.
struct SettlingState {
    std::uint32_t generation=0,quietTicks=0,unsupportedTicks=0,candidate=0;
    bool operator==(const SettlingState&) const = default;
};
static_assert(sizeof(SettlingState)==16);
struct SettlingObservation {
    std::uint32_t generation=0;
    float speed=0;
    bool eligible=false,enterContact=false,holdContact=false,completedTick=false;
};
inline void validateSettlingProfile(const SettlingProfile& p) {
    if(!std::isfinite(p.enterSpeed) || !std::isfinite(p.exitSpeed) ||
       p.enterSpeed<0 || p.exitSpeed<=p.enterSpeed || !p.acquireTicks || !p.releaseTicks)
        throw std::invalid_argument("Invalid settling hysteresis profile");
}
// Call once per completed fixed tick, never per render frame or substep. A fault
// invalidates history; it cannot create deposition eligibility from stale state.
// Contact acquisition must be a subset of the looser retention contact test.
inline SettlingState advanceSettling(SettlingState state,const SettlingObservation& o,
                                    const SettlingProfile& p={}) {
    validateSettlingProfile(p);
    if(o.enterContact && !o.holdContact)
        throw std::invalid_argument("Settling acquisition contact must imply retention contact");
    const SettlingState empty{o.generation,0,0,0};
    if(!o.completedTick || !o.eligible || !o.generation || !std::isfinite(o.speed) || o.speed<0)
        return empty;
    if(state.generation!=o.generation) state=empty;
    // Reject impossible histories, including those produced under another profile.
    if(state.candidate>1 || state.quietTicks>p.acquireTicks || state.unsupportedTicks>=p.releaseTicks ||
       (state.candidate && state.quietTicks!=p.acquireTicks) ||
       (!state.candidate && (state.unsupportedTicks || state.quietTicks==p.acquireTicks)))
        state=empty;
    if(state.candidate) {
        if(o.speed>p.exitSpeed) return empty;
        if(o.holdContact) state.unsupportedTicks=0;
        else if(state.unsupportedTicks==p.releaseTicks-1) return empty;
        else ++state.unsupportedTicks;
    } else if(o.speed<=p.enterSpeed && o.enterContact) {
        // The validated upper bound makes this safe even with UINT32_MAX dwell.
        ++state.quietTicks;
        if(state.quietTicks==p.acquireTicks) state.candidate=1;
    } else state=empty;
    return state;
}
}
