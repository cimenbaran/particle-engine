#include "Scenes.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
namespace pw {
namespace {
std::uint32_t integer(const nlohmann::json& value,std::uint32_t maximum,const char* field) {
    if(!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>()<0) ||
       value.get<std::uint64_t>()>maximum) throw std::runtime_error(std::string("Invalid scene ")+field);
    return value.get<std::uint32_t>();
}
}
ScenePreset parseScene(std::string_view text) {
    if(text.size()>65536) throw std::runtime_error("Scene exceeds 64 KiB");
    const auto json=nlohmann::json::parse(text);
    if(!json.is_object())throw std::runtime_error("Scene must be an object");
    const auto version=integer(json.at("schema_version"),2,"schema version");
    if(version!=1 && version!=2)throw std::runtime_error("Unsupported scene version");
    std::set<std::string> fields{"schema_version","name","seed","count","minimum","maximum","materials"};
    if(version==2)fields.insert("sphere");
    if(json.size()!=fields.size()) throw std::runtime_error("Scene requires exactly the versioned fields");
    for(const auto& [key,value]:json.items()) {if(!fields.contains(key)) throw std::runtime_error("Unknown scene field: "+key);}

    ScenePreset scene;
    if(version==2) {
        const auto& sphere=json.at("sphere");
        if(!sphere.is_object() || sphere.size()!=2 || !sphere.contains("centre") || !sphere.contains("radius") ||
           !sphere.at("centre").is_array() || sphere.at("centre").size()!=3 || !sphere.at("radius").is_number())
            throw std::runtime_error("Scene sphere requires numeric centre and radius");
        for(const auto& value:sphere.at("centre"))if(!value.is_number())throw std::runtime_error("Scene sphere centre must be numeric");
        scene.obstacle={sphere.at("centre").get<std::array<float,3>>(),sphere.at("radius").get<float>()};
        validateAnalyticSphere(scene.obstacle);
        if(scene.obstacle.radius==0)throw std::runtime_error("Version-2 scene requires an enabled sphere");
    }
    scene.name=json.at("name").get<std::string>();
    if(scene.name.empty() || scene.name.size()>80) throw std::runtime_error("Scene name must be 1-80 bytes");
    scene.seed=integer(json.at("seed"),UINT32_MAX,"seed");
    if(json.at("count").is_string()) {
        if(json.at("count")!="capacity") throw std::runtime_error("Scene count string must be capacity");
    } else scene.count=integer(json.at("count"),1048576,"count");
    for(const auto* key:{"minimum","maximum"}) if(!json.at(key).is_array() || json.at(key).size()!=3)
        throw std::runtime_error("Scene bounds must contain three numbers");
    for(const auto* key:{"minimum","maximum"}) for(const auto& coordinate:json.at(key))
        if(!coordinate.is_number()) throw std::runtime_error("Scene bounds must be numeric");
    scene.minimum=json.at("minimum").get<std::array<float,3>>();scene.maximum=json.at("maximum").get<std::array<float,3>>();
    for(unsigned axis=0;axis<3;++axis) {
        const float low=axis==1?.05f:-7.95f,high=axis==1?64.0f:7.95f;
        if(!std::isfinite(scene.minimum[axis]) || !std::isfinite(scene.maximum[axis]) ||
           scene.minimum[axis]<low || scene.maximum[axis]>high || scene.minimum[axis]>scene.maximum[axis])
            throw std::runtime_error("Scene bounds must fit the bounded world and remain ordered");
    }
    scene.materials=json.at("materials").get<std::vector<std::string>>();
    if(scene.materials.empty() || scene.materials.size()>64) throw std::runtime_error("Scene requires 1-64 solid materials");
    std::set<std::string> unique;
    for(const auto& name:scene.materials) if(name.empty() || name.size()>80 || !unique.insert(name).second)
        throw std::runtime_error("Scene material names must be nonempty and unique");
    if(scene.obstacle.radius>0) {
        double nearest=0;
        for(unsigned a=0;a<3;++a){double d=double(scene.obstacle.centre[a])-std::clamp(scene.obstacle.centre[a],scene.minimum[a],scene.maximum[a]);nearest+=d*d;}
        const double radius=double(scene.obstacle.radius)+0.0625;
        if(nearest<=radius*radius)throw std::runtime_error("Scene spawn bounds intersect sphere terrain");
        for(unsigned a=0;a<3;++a)if(scene.minimum[a]<(a==1?0.0625f:-7.9375f) || scene.maximum[a]>(a==1?15.9375f:7.9375f))
            throw std::runtime_error("Sphere scene must fit the granular enclosure");
    }
    return scene;
}
std::string encodeScene(const ScenePreset& scene) {
    validateAnalyticSphere(scene.obstacle);
    nlohmann::json json={{"schema_version",scene.obstacle.radius>0?2:1},{"name",scene.name},{"seed",scene.seed},
        {"count",scene.count?nlohmann::json(*scene.count):nlohmann::json("capacity")},
        {"minimum",scene.minimum},{"maximum",scene.maximum},{"materials",scene.materials}};
    if(scene.obstacle.radius>0)json["sphere"]={{"centre",scene.obstacle.centre},{"radius",scene.obstacle.radius}};
    const auto text=json.dump(2);parseScene(text);return text;
}
ScenePreset loadScene(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()<0 || file.tellg()>65536) throw std::runtime_error("Cannot read scene or file exceeds 64 KiB: "+path.string());
    std::string text(static_cast<std::size_t>(file.tellg()),'\0');file.seekg(0);file.read(text.data(),std::streamsize(text.size()));
    if(!file) throw std::runtime_error("Cannot read complete scene: "+path.string());
    return parseScene(text);
}
std::vector<std::uint32_t> sceneMaterials(const ScenePreset& scene,const MaterialLibrary& library) {
    std::vector<std::uint32_t> ids;
    for(const auto& name:scene.materials) {
        const auto found=std::find_if(library.entries.begin(),library.entries.end(),[&](const auto& material){return material.name==name;});
        if(found==library.entries.end() || found->gpu.identity[1]!=0) throw std::runtime_error("Scene requires a known solid material: "+name);
        ids.push_back(std::uint32_t(found-library.entries.begin()));
    }
    if(ids.empty()) throw std::runtime_error("Scene requires a solid material");return ids;
}
void validateSceneCapacity(const ScenePreset& scene,std::uint32_t capacity) {
    if(!capacity || capacity>1048576 || scene.count.value_or(capacity)>capacity) throw std::runtime_error("Scene particle count exceeds the selected capacity");
}
std::vector<ParticleGpu> instantiateScene(const ScenePreset& scene,const MaterialLibrary& library,std::uint32_t capacity,
                                        std::span<const std::uint32_t> materials,std::optional<std::uint32_t> countOverride) {
    validateSceneCapacity(scene,capacity);
    if(materials.empty()) throw std::runtime_error("Scene requires material IDs");
    for(auto id:materials) if(id>=library.entries.size() || library.entries[id].gpu.identity[1]!=0) throw std::runtime_error("Invalid scene material ID");
    const auto count=countOverride.value_or(scene.count.value_or(capacity));
    if(count>capacity) throw std::runtime_error("Scene override exceeds capacity");
    std::vector<ParticleGpu> initial(capacity);
    std::mt19937 random(scene.seed);
    auto unit=[&]{return float(random()>>8)/16777216.0f;};
    for(std::uint32_t id=0;id<capacity;++id) initial[id].identity[2]=id;
    for(std::uint32_t id=0;id<count;++id) {
        auto& particle=initial[id];
        for(unsigned axis=0;axis<3;++axis) particle.positionRadius[axis]=scene.minimum[axis]+unit()*(scene.maximum[axis]-scene.minimum[axis]);
        particle.positionRadius[3]=.045f;
        const auto material=materials[id%materials.size()];
        const float radius=particle.positionRadius[3];
        particle.velocityInvMass[3]=1.0f/(4.0f/3.0f*3.14159265f*radius*radius*radius*library.entries[material].gpu.mechanical[0]);
        particle.thermalState[0]=293.15f;particle.identity[0]=material;particle.identity[1]=1;particle.identity[3]=1;
    }
    return initial;
}
}
