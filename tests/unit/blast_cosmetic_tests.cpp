#include "core/BlastCosmetics.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#define CHECK(x) do{if(!(x))throw std::runtime_error("Check failed: " #x);}while(false)
int main(){try{
    using namespace pw;
    BlastCosmeticEvent event{{-8,3.875f,-4},.5f,7};
    auto rejects=[](auto f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}CHECK(bad);};
    unsigned seeds=0;
    for(auto sequence:{0ull,1ull,7ull,0xffffffffffffffffull})for(float radius:{.001f,.125f,.5f,4.f})
    for(auto centre:{std::array<float,3>{0,0,0},std::array<float,3>{-16,16,16}}) {
        event={centre,radius,sequence};const auto batch=planBlastCosmetics(event,{128,64});
        CHECK(batch.size()==192 && batch==planBlastCosmetics(event,{128,64}));
        const auto dust=planBlastCosmetics(event,{128,0}),spark=planBlastCosmetics(event,{0,64});
        CHECK(std::equal(dust.begin(),dust.end(),batch.begin()) && std::equal(spark.begin(),spark.end(),batch.begin()+128));
        for(unsigned i=0;i<batch.size();++i){const auto& p=batch[i];double distance=0,speed=0;
            for(unsigned a=0;a<3;++a){CHECK(std::isfinite(p.position[a]) && std::isfinite(p.velocity[a]));distance+=std::pow(double(p.position[a])-centre[a],2);speed+=double(p.velocity[a])*p.velocity[a];}
            CHECK(std::sqrt(distance)<=radius*.5+2e-6 && p.velocity[1]>=0);
            CHECK(p.position[1]>=centre[1] && p.lifetime>0 && p.radius>0);
            if(i<128){CHECK(p.kind==BlastCosmeticKind::Dust && speed>=.25-1e-6 && speed<=4.00001 && p.lifetime>=.8f && p.lifetime<=2.f);}
            else {CHECK(p.kind==BlastCosmeticKind::Spark && speed>=9-1e-5 && speed<=64.0001 && p.lifetime>=.2f && p.lifetime<=.8f);}
            ++seeds;
        }
        CHECK(planBlastCosmetics(event,{0,0}).empty());
    }
    event={{0,1,0},.5f,0};const auto first=planBlastCosmetics(event);event.sequence=1;CHECK(first!=planBlastCosmetics(event));
    rejects([&]{planBlastCosmetics(event,{129,0});});rejects([&]{planBlastCosmetics(event,{0,65});});
    for(float bad:{0.f,-1.f,4.01f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){auto e=event;e.radius=bad;rejects([&]{planBlastCosmetics(e);});}
    for(unsigned a=0;a<3;++a)for(float bad:{-17.f,17.f,std::numeric_limits<float>::quiet_NaN()}){auto e=event;e.centre[a]=bad;rejects([&]{planBlastCosmetics(e);});}
    auto below=event;below.centre[1]=-.01f;rejects([&]{planBlastCosmetics(below);});
    std::cout<<"Cosmetic blast plans passed: "<<seeds<<" bounded seeds, independent quotas, reproducible streams and invalid inputs\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
