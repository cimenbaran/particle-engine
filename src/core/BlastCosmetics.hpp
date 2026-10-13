#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace pw {
// Visual-only descriptors: never particle mass, terrain products or energy charges.
struct BlastCosmeticEvent {
    std::array<float,3> centre{};
    float radius=.5f;
    std::uint64_t sequence=0;
};
enum class BlastCosmeticKind : std::uint32_t {Dust,Spark};
struct BlastCosmeticSeed {
    std::array<float,3> position{},velocity{};
    float radius=0,lifetime=0;
    BlastCosmeticKind kind=BlastCosmeticKind::Dust;
    bool operator==(const BlastCosmeticSeed&) const=default;
};
struct BlastCosmeticQuota {
    std::uint32_t dust=48,sparks=16;
    static constexpr std::uint32_t maxDust=128,maxSparks=64;
};
inline std::vector<BlastCosmeticSeed> planBlastCosmetics(const BlastCosmeticEvent& event,BlastCosmeticQuota quota={}) {
    if(!std::isfinite(event.radius) || event.radius<=0 || event.radius>4 ||
       quota.dust>BlastCosmeticQuota::maxDust || quota.sparks>BlastCosmeticQuota::maxSparks)
        throw std::invalid_argument("Invalid cosmetic blast radius or quota");
    for(unsigned a=0;a<3;++a)if(!std::isfinite(event.centre[a]) || event.centre[a]<(a==1?0:-16) || event.centre[a]>16)
        throw std::invalid_argument("Cosmetic blast centre outside basin");
    // SplitMix64 with defined unsigned wrap; no global RNG or physical state.
    std::uint64_t random=event.sequence;
    const auto unit=[&]() {
        auto z=(random+=0x9e3779b97f4a7c15ull);
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;z^=z>>31;
        return double(z>>11)*0x1p-53;
    };
    std::vector<BlastCosmeticSeed> result;result.reserve(quota.dust+quota.sparks);
    for(unsigned kind=0;kind<2;++kind) {
        // Each visual family has its own random stream, unaffected by the other quota.
        random=event.sequence^(kind?0xd1b54a32d192ed03ull:0);
        for(unsigned i=0;i<(kind?quota.sparks:quota.dust);++i) {
            const double y=unit(),angle=unit()*6.2831853071795864769;
            const double horizontal=std::sqrt(1-y*y);
            const std::array<double,3> direction{horizontal*std::cos(angle),y,horizontal*std::sin(angle)};
            const double distance=event.radius*.5*std::cbrt(unit()),speed=(kind?3.:.5)+unit()*(kind?5.:1.5);
            BlastCosmeticSeed seed;seed.kind=kind?BlastCosmeticKind::Spark:BlastCosmeticKind::Dust;
            seed.radius=kind?.018f:.055f;seed.lifetime=float((kind?.2:.8)+unit()*(kind?.6:1.2));
            for(unsigned a=0;a<3;++a){seed.position[a]=float(double(event.centre[a])+direction[a]*distance);seed.velocity[a]=float(direction[a]*speed);}
            result.push_back(seed);
        }
    }
    return result;
}
}
