#pragma once
#include <algorithm>
#include <cstdint>
namespace pw {
// Negative backend values mean unavailable, not zero usage.
struct ReportedMemory {
    std::int64_t gpu=-1,textures=-1,renderTargets=-1,transientVertices=-1,transientIndices=-1;
};
class ReportedMemoryPeaks {
public:
    void observe(const ReportedMemory& sample) {
        auto keep=[](std::int64_t& peak,std::int64_t value) {if(value>=0) peak=std::max(peak,value);};
        keep(peaks_.gpu,sample.gpu);keep(peaks_.textures,sample.textures);keep(peaks_.renderTargets,sample.renderTargets);
        keep(peaks_.transientVertices,sample.transientVertices);keep(peaks_.transientIndices,sample.transientIndices);
        ++observations_;
    }
    void reset() {peaks_={};observations_=0;}
    const ReportedMemory& values() const {return peaks_;}
    std::uint64_t observations() const {return observations_;}
private:
    ReportedMemory peaks_;
    std::uint64_t observations_=0;
};
}
