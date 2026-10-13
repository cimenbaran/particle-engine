#include "assets/Scenes.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do {if(!(x)) throw std::runtime_error("Check failed: " #x);} while(false)
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);const std::filesystem::path assets=argv[1];
        const auto library=pw::loadMaterials(assets/"materials/default.json");
        const auto scene=pw::loadScene(assets/"scenes/ballistic.json");
        const auto ids=pw::sceneMaterials(scene,library);
        const auto first=pw::instantiateScene(scene,library,257,ids);
        const auto second=pw::instantiateScene(scene,library,257,ids);
        CHECK(first.size()==257 && std::memcmp(first.data(),second.data(),first.size()*sizeof(pw::ParticleGpu))==0);
        for(unsigned i=0;i<first.size();++i) {
            const auto& particle=first[i];CHECK(particle.identity[2]==i && particle.identity[3]==1 && particle.identity[1]==1);
            CHECK(particle.identity[0]==ids[i%ids.size()]);
            for(unsigned axis=0;axis<3;++axis) CHECK(particle.positionRadius[axis]>=scene.minimum[axis] && particle.positionRadius[axis]<=scene.maximum[axis]);
        }
        auto changed=scene;++changed.seed;
        const auto another=pw::instantiateScene(changed,library,257,ids);
        CHECK(std::memcmp(first.data(),another.data(),first.size()*sizeof(pw::ParticleGpu))!=0);
        const auto empty=pw::loadScene(assets/"scenes/empty.json");
        const auto emptyParticles=pw::instantiateScene(empty,library,129,pw::sceneMaterials(empty,library));
        for(unsigned i=0;i<emptyParticles.size();++i) CHECK(emptyParticles[i].identity[1]==0 && emptyParticles[i].identity[2]==i);
        const auto sand=pw::loadScene(assets/"scenes/sand_drop.json");
        const auto sandIds=pw::sceneMaterials(sand,library);CHECK(sandIds.size()==1 && library.entries[sandIds.front()].name=="Sand");
        const auto partial=pw::instantiateScene(sand,library,8192,sandIds);
        CHECK(partial[4095].identity[1]==1 && partial[4096].identity[1]==0 && partial[8191].identity[2]==8191);
        bool overCapacity=false;try{pw::validateSceneCapacity(sand,4095);}catch(const std::exception&){overCapacity=true;}CHECK(overCapacity);
        std::ifstream file(assets/"scenes/ballistic.json");nlohmann::json source;file>>source;
        auto rejects=[&](auto change) {
            auto candidate=source;change(candidate);bool failed=false;
            try {const auto parsed=pw::parseScene(candidate.dump());pw::sceneMaterials(parsed,library);}catch(const std::exception&){failed=true;}
            if(!failed) throw std::runtime_error("Accepted invalid scene: "+candidate.dump());
        };
        rejects([](auto& j){j["schema_version"]=2;});
        rejects([](auto& j){j["seed"]=-1;});rejects([](auto& j){j["seed"]=4294967296ull;});
        rejects([](auto& j){j["seed"]=1.5;});rejects([](auto& j){j["count"]=1048577;});
        rejects([](auto& j){j["count"]="all";});rejects([](auto& j){j["name"]="";});
        rejects([](auto& j){j["minimum"][0]=-8;});rejects([](auto& j){j["minimum"][1]=0;});
        rejects([](auto& j){j["minimum"][0]=7;});rejects([](auto& j){j["maximum"][1]=65;});
        rejects([](auto& j){j["minimum"]={1,2};});rejects([](auto& j){j["minimum"][0]=true;});
        rejects([](auto& j){j["materials"]={"Missing"};});rejects([](auto& j){j["materials"]={"Water"};});
        rejects([](auto& j){j["materials"]={"Stone","Stone"};});rejects([](auto& j){j["materials"]=nlohmann::json::array();});
        rejects([](auto& j){j["unrecognized"]=true;});rejects([](auto& j){j.erase("seed");});
        const auto sphere=pw::loadScene(assets/"scenes/sphere_drop.json");
        const auto roundtrip=pw::parseScene(pw::encodeScene(sphere));
        CHECK(roundtrip.obstacle.radius==1.25f && roundtrip.obstacle.centre==sphere.obstacle.centre);
        const auto sphereParticles=pw::instantiateScene(sphere,library,256,pw::sceneMaterials(sphere,library));
        for(const auto& body:sphereParticles){double distance=0;for(unsigned a=0;a<3;++a){double d=body.positionRadius[a]-sphere.obstacle.centre[a];distance+=d*d;}CHECK(distance>1.3125*1.3125);}
        source=nlohmann::json::parse(pw::encodeScene(sphere));
        rejects([](auto& j){j["sphere"]["radius"]=-1;});rejects([](auto& j){j["sphere"]["radius"]=0;});
        rejects([](auto& j){j["sphere"]["radius"]=true;});rejects([](auto& j){j["sphere"]["centre"]={0,0,0};});
        rejects([](auto& j){j["sphere"]["centre"][0]=true;});rejects([](auto& j){j["sphere"]["extra"]=1;});
        rejects([](auto& j){j["minimum"]={-1,2,-1};});rejects([](auto& j){j["maximum"][1]=20;});
        rejects([](auto& j){j.erase("sphere");});rejects([](auto& j){j["schema_version"]=1;});
        std::cout<<"Scene schema, material resolution, capacity limits, seeded repeatability and inactive slots passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
