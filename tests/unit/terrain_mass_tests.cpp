#include "core/TerrainMass.hpp"
#include <iostream>
#include <random>
#define CHECK(v) do{if(!(v))throw std::runtime_error("Terrain mass check failed: " #v);}while(false)
using namespace pw::terrain;
template<class Exception,class F>void rejects(F f){bool rejected=false;try{f();}catch(const Exception&){rejected=true;}CHECK(rejected);}
int main() {
    std::uint64_t cases=0;
    auto check=[&](std::uint32_t count,float density,bool deposit,double tolerance) {
        const auto profile=conversionMassProfile(density);
        const auto plan=deposit?planGrainDeposit(count,density,tolerance):planVoxelDebris(count,density,tolerance);
        // Independent long-double nearest-mass oracle over both neighboring
        // integer counts, using the inverse-mass bits actually stored on GPU.
        const long double voxel=static_cast<long double>(density)/512;
        const long double grain=1/static_cast<long double>(plan.inverseGrainMass);
        const long double source=count*(deposit?grain:voxel),unit=deposit?voxel:grain;
        const auto lower=static_cast<std::uint64_t>(std::floor(source/unit));
        const auto upper=lower+1;
        const long double loError=std::abs(lower*unit-source),hiError=std::abs(upper*unit-source);
        const auto nearest=hiError<=loError?upper:lower;
        CHECK((deposit?plan.voxels:plan.grains)==nearest);
        CHECK((deposit?plan.grains:plan.voxels)==count);
        const long double destination=nearest*unit,delta=destination-source;
        CHECK(std::abs(static_cast<long double>(plan.sourceKilograms)-source)<=1e-14L*std::max(1.0L,source));
        CHECK(std::abs(static_cast<long double>(plan.destinationKilograms)-destination)<=1e-14L*std::max(1.0L,destination));
        CHECK(std::abs(static_cast<long double>(plan.quantizationKilograms)-delta)<=1e-14L*std::max(1.0L,source));
        CHECK(std::abs((plan.sourceKilograms+plan.quantizationKilograms)-plan.destinationKilograms)<=4*std::numeric_limits<double>::epsilon()*std::max(1.0,plan.destinationKilograms));
        CHECK(plan.admissible==(std::abs(delta)<=tolerance*source && (source==0 || destination>0)));
        CHECK(std::isfinite(profile.inverseMass) && profile.inverseMass>0);
        // Encoded masses remain density-derived to float reciprocal precision.
        const long double nominal=4.0L/3*std::numbers::pi_v<long double>*.0625L*.0625L*.0625L*density;
        CHECK(std::abs(grain-nominal)/nominal<7e-8L);
        if(!deposit) {
            const auto back=planGrainDeposit(plan.grains,density);
            CHECK(back.voxels==count && back.admissible);
            CHECK(plan.quantizationKilograms+back.quantizationKilograms==0);
        }
        ++cases;
    };
    for(float density:{.001f,1.0f,997.0f,1600.0f,2700.0f,100000.0f}) {
        for(unsigned n=0;n<=10000;++n){check(n,density,false,.05);check(n,density,true,.05);}
        for(auto n:{65535u,65536u,65537u,1048576u,8388608u,1000000000u}){check(n,density,false,.05);check(n,density,true,.05);}
        const auto one=planVoxelDebris(1,density);CHECK(one.grains==2 && one.admissible && one.relativeError>.04719 && one.relativeError<.04721);
        const auto single=planGrainDeposit(1,density);CHECK(single.voxels==1 && !single.admissible && single.quantizationKilograms>0);
        const auto pair=planGrainDeposit(2,density);CHECK(pair.voxels==1 && pair.admissible && pair.quantizationKilograms<0);
        CHECK(one.destinationKilograms==pair.sourceKilograms && one.sourceKilograms==pair.destinationKilograms);
        CHECK(one.quantizationKilograms+pair.quantizationKilograms==0);
        CHECK(!planVoxelDebris(1,density,.01).admissible);
        CHECK(planVoxelDebris(0,density,0).admissible && planGrainDeposit(0,density,0).admissible);
        rejects<std::overflow_error>([&]{planVoxelDebris(UINT32_MAX,density);});
        check(UINT32_MAX,density,true,.05);
    }
    // Exact rational binary32 products, independently derived with integer
    // arithmetic: just below, exactly on, and just above half a product count.
    CHECK(planVoxelDebris(253061881,2700).grains==483312608);
    CHECK(planVoxelDebris(134217728,2700).grains==256336988);
    CHECK(planVoxelDebris(15373575,2700).grains==29361367);
    CHECK(planGrainDeposit(142849177,2700).voxels==74795651);
    CHECK(planGrainDeposit(369824798,2700).voxels==193639805);
    const auto tinyLoss=planVoxelDebris(149591303,2700,0);
    const auto tinyGain=planVoxelDebris(118844153,2700,0);
    CHECK(!tinyLoss.admissible && tinyLoss.quantizationKilograms<0);
    CHECK(!tinyGain.admissible && tinyGain.quantizationKilograms>0);
    CHECK(planVoxelDebris(268435456,2700,0).admissible);
    std::mt19937 random(902144);
    for(unsigned i=0;i<4096;++i) {
        const float density=.001f+float(random()%100000000)/1000.0f;
        const auto count=random()%8388609;
        const double tolerance=double(random()%5001)/100000;
        check(count,density,false,tolerance);check(count,density,true,tolerance);
    }
    for(float density:{0.0f,-1.0f,.0001f,100001.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        rejects<std::invalid_argument>([&]{planVoxelDebris(0,density);});
    for(double tolerance:{-.001,.050001,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        rejects<std::invalid_argument>([&]{planGrainDeposit(0,1600,tolerance);});
    std::cout<<"Terrain mass planning passed: "<<cases<<" balances, closest-count oracle, explicit quantization, round trips, strict tolerances, invalid inputs and overflow\n";
}
