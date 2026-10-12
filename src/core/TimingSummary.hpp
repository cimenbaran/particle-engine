#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
namespace pw {
struct TimingSummary {double minimum{},median{},p95{},maximum{},mean{};};
inline TimingSummary summarize(std::span<const double> samples) {
    if(samples.empty()) throw std::runtime_error("Cannot summarize empty samples");
    std::vector<double> sorted(samples.begin(),samples.end());double sum=0;
    for(auto sample:sorted) {if(!std::isfinite(sample) || sample<0) throw std::runtime_error("Invalid timing sample");sum+=sample;}
    std::sort(sorted.begin(),sorted.end());
    auto percentile=[&](double fraction) {return sorted[std::size_t(std::ceil(fraction*sorted.size()))-1];};
    return {sorted.front(),percentile(.5),percentile(.95),sorted.back(),sum/sorted.size()};
}
}
