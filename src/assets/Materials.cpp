#include "Materials.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <stdexcept>

namespace pw {
namespace {
using Json=nlohmann::json;
void keys(const Json& object, std::initializer_list<std::string_view> allowed) {
    if(!object.is_object() || object.size()!=allowed.size()) throw std::runtime_error("Material object has missing or extra fields");
    for(auto key:allowed) if(!object.contains(key)) throw std::runtime_error("Missing material field: "+std::string(key));
}
double preciseScalar(const Json& object, const char* key, double low, double high) {
    const auto& field=object.at(key);
    if(!field.is_number()) throw std::runtime_error(std::string(key)+" must be numeric");
    const double value=field.get<double>();
    if(!std::isfinite(value) || value<low || value>high) throw std::runtime_error(std::string(key)+" is outside its supported range");
    return value;
}
float scalar(const Json& object, const char* key, double low, double high) {
    return static_cast<float>(preciseScalar(object,key,low,high));
}
std::uint32_t index(const Json& field) {
    if(!field.is_number_integer() || field.get<double>()<0 || field.get<double>()>255)
        throw std::runtime_error("Material IDs must be integers in [0,255]");
    return field.get<std::uint32_t>();
}
void append32(std::vector<std::byte>& bytes, std::uint32_t value) {
    for(unsigned shift=0;shift<32;shift+=8) bytes.push_back(static_cast<std::byte>((value>>shift)&255));
}
void append64(std::vector<std::byte>& bytes, std::uint64_t value) {
    append32(bytes,static_cast<std::uint32_t>(value)); append32(bytes,static_cast<std::uint32_t>(value>>32));
}
}
std::uint64_t fingerprint(std::string_view text) {
    // FNV-1a is a stable compatibility fingerprint, not a security signature.
    std::uint64_t value=14695981039346656037ull;
    for(unsigned char byte:text) {value^=byte;value*=1099511628211ull;}
    return value;
}
std::vector<MaterialGpu> MaterialLibrary::packed() const {
    std::vector<MaterialGpu> result; result.reserve(entries.size());
    for(const auto& entry:entries) result.push_back(entry.gpu);
    return result;
}
MaterialLibrary parseMaterials(std::string_view text) {
    if(text.size()>1024*1024) throw std::runtime_error("Material JSON exceeds 1 MiB limit");
    const auto root=Json::parse(text);
    keys(root,{"schema_version","units","materials"});
    if(!root.at("schema_version").is_number_integer() || root.at("schema_version")!=2)
        throw std::runtime_error("Unsupported material schema version");
    const Json units={{"length","m"},{"mass","kg"},{"time","s"},{"temperature","K"}};
    if(root.at("units")!=units) throw std::runtime_error("Material units must be m, kg, s and K");
    const auto& materials=root.at("materials");
    if(!materials.is_array() || materials.empty() || materials.size()>256) throw std::runtime_error("Expected 1-256 materials");
    MaterialLibrary result;
    std::set<std::string> names;
    for(const auto& item:materials) {
        keys(item,{"id","name","phase","density","friction","restitution","drag","conductivity","heat_capacity","ignition_kelvin","burn_rate","color","transitions","fracture_resistance"});
        MaterialDesc desc;
        desc.fractureResistance=preciseScalar(item,"fracture_resistance",1e-6,1e12);
        desc.name=item.at("name").get<std::string>();
        if(desc.name.empty() || desc.name.size()>64 || !names.insert(desc.name).second)
            throw std::runtime_error("Material names must be unique and 1-64 bytes long");
        auto& gpu=desc.gpu;
        gpu.identity[0]=index(item.at("id"));
        const auto phase=item.at("phase").get<std::string>();
        if(phase=="solid") gpu.identity[1]=0;
        else if(phase=="liquid") gpu.identity[1]=1;
        else if(phase=="gas") gpu.identity[1]=2;
        else throw std::runtime_error("Unsupported material phase: "+phase);
        gpu.identity[2]=gpu.identity[1];
        gpu.mechanical[0]=scalar(item,"density",0.001,100000);
        gpu.mechanical[1]=scalar(item,"friction",0,1);
        gpu.mechanical[2]=scalar(item,"restitution",0,1);
        gpu.mechanical[3]=scalar(item,"drag",0,100);
        gpu.thermal[0]=scalar(item,"conductivity",0,10000);
        gpu.thermal[1]=scalar(item,"heat_capacity",0.001,1000000);
        gpu.thermal[2]=scalar(item,"ignition_kelvin",0,100000);
        gpu.thermal[3]=scalar(item,"burn_rate",0,100);
        if(gpu.thermal[3]>0 && gpu.thermal[2]==0) throw std::runtime_error("Burning material needs a positive ignition threshold");
        const auto& color=item.at("color");
        if(!color.is_array() || color.size()!=4) throw std::runtime_error("Material color requires four normalized components");
        for(unsigned i=0;i<4;++i) {
            const auto value=color[i].get<double>();
            if(!color[i].is_number() || !std::isfinite(value) || value<0 || value>1) throw std::runtime_error("Color is outside [0,1]");
            gpu.color[i]=static_cast<float>(value);
        }
        if(!item.at("transitions").is_array()) throw std::runtime_error("Transitions must be an array of IDs");
        std::set<std::uint32_t> targets;
        for(const auto& target:item.at("transitions")) {
            const auto id=index(target);
            if(!targets.insert(id).second) throw std::runtime_error("Duplicate transition target");
            desc.transitions.push_back(id);
        }
        result.entries.push_back(std::move(desc));
    }
    std::sort(result.entries.begin(),result.entries.end(),[](const auto& a,const auto& b){return a.gpu.identity[0]<b.gpu.identity[0];});
    for(std::size_t i=0;i<result.entries.size();++i) {
        if(result.entries[i].gpu.identity[0]!=i) throw std::runtime_error("Material IDs must be unique and contiguous from zero");
        for(auto target:result.entries[i].transitions) if(target>=result.entries.size()) throw std::runtime_error("Unknown material transition target");
    }
    std::vector<unsigned> visit(result.entries.size());
    std::function<void(std::uint32_t)> walk=[&](std::uint32_t id) {
        if(visit[id]==1) throw std::runtime_error("Cyclic material transitions are not supported");
        if(visit[id]==2) return;
        visit[id]=1;
        for(auto target:result.entries[id].transitions) walk(target);
        visit[id]=2;
    };
    for(std::uint32_t i=0;i<result.entries.size();++i) walk(i);
    result.contentHash=fingerprint(root.dump());
    return result;
}
MaterialLibrary loadMaterials(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input) throw std::runtime_error("Cannot open material JSON: "+path.string());
    const auto size=input.tellg();
    if(size<0 || size>1024*1024) throw std::runtime_error("Material JSON exceeds 1 MiB limit");
    std::string text(static_cast<std::size_t>(size),'\0');
    input.seekg(0); input.read(text.data(),static_cast<std::streamsize>(text.size()));
    if(!input) throw std::runtime_error("Cannot read material JSON");
    return parseMaterials(text);
}
void validateMaterialReload(const MaterialLibrary& current,const MaterialLibrary& candidate,std::span<const std::uint32_t> liveIds) {
    for(auto id:liveIds) {
        if(id>=current.entries.size() || id>=candidate.entries.size() ||
           current.entries[id].name!=candidate.entries[id].name ||
           current.entries[id].gpu.identity[1]!=candidate.entries[id].gpu.identity[1])
            throw std::runtime_error("A live material was removed, renamed or changed phase; restart the app to apply this edit");
        if(current.entries[id].gpu.mechanical[0]!=candidate.entries[id].gpu.mechanical[0])
            throw std::runtime_error("Changing live material density requires restarting the app to recalculate particle masses");
    }
}
std::vector<std::byte> encodeMaterials(const MaterialLibrary& library) {
    std::vector<std::byte> bytes;
    append32(bytes,0x544d5750); // ASCII PWMT
    append32(bytes,2); append32(bytes,72); append32(bytes,static_cast<std::uint32_t>(library.entries.size()));
    append64(bytes,fingerprint(materialSchema)); append64(bytes,library.contentHash);
    for(const auto& entry:library.entries) {
        for(float value:entry.gpu.mechanical) append32(bytes,std::bit_cast<std::uint32_t>(value));
        for(float value:entry.gpu.thermal) append32(bytes,std::bit_cast<std::uint32_t>(value));
        for(float value:entry.gpu.color) append32(bytes,std::bit_cast<std::uint32_t>(value));
        for(auto value:entry.gpu.identity) append32(bytes,value);
        append64(bytes,std::bit_cast<std::uint64_t>(entry.fractureResistance));
    }
    return bytes;
}
}
