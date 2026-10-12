#include "assets/Materials.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <bit>
using Json=nlohmann::json;
#define CHECK(x) do {if(!(x)) throw std::runtime_error("Check failed: " #x);} while(false)
int main(int argc,char** argv) {
 try {
    CHECK(argc==2);
    std::ifstream in(argv[1]); Json source;in>>source;
    const auto library=pw::parseMaterials(source.dump());
    CHECK(library.entries.size()==7 && library.entries[1].name=="Sand");
    const auto bytes=pw::encodeMaterials(library);
    CHECK(bytes.size()==32+7*64);
    CHECK(bytes[0]==std::byte{'P'} && bytes[1]==std::byte{'W'} && bytes[2]==std::byte{'M'} && bytes[3]==std::byte{'T'});
    CHECK(bytes[8]==std::byte{64} && bytes[12]==std::byte{7});
    const auto expectedBits=std::bit_cast<std::uint32_t>(2600.f);
    for(unsigned i=0;i<4;++i) CHECK(bytes[32+i]==std::byte((expectedBits>>(8*i))&255));
    CHECK(pw::parseMaterials(source.dump(4)).contentHash==library.contentHash);
    auto rejects=[&](auto change){auto candidate=source;change(candidate);bool failed=false;try{pw::parseMaterials(candidate.dump());}catch(const std::exception&){failed=true;}CHECK(failed);};
    rejects([](auto& j){j["schema_version"]=2;});
    rejects([](auto& j){j["units"]["temperature"]="C";});
    rejects([](auto& j){j["materials"][0]["heat_capacity"]=0;});
    rejects([](auto& j){j["materials"][0]["drag"]=-1;});
    rejects([](auto& j){j["materials"][0]["restitution"]=1.01;});
    rejects([](auto& j){j["materials"][1]["id"]=0;});
    rejects([](auto& j){j["materials"][0]["transitions"]={255};});
    rejects([](auto& j){j["materials"][0]["transitions"]={1};j["materials"][1]["transitions"]={0};});
    rejects([](auto& j){j["materials"][0]["color"][0]=2;});
    rejects([](auto& j){j["materials"][0]["typo"]=true;});
    const std::uint32_t live[]{1};
    auto edited=source;edited["materials"][1]["drag"]=1;
    auto candidate=pw::parseMaterials(edited.dump());
    pw::validateMaterialReload(library,candidate,live);
    CHECK(candidate.contentHash!=library.contentHash);
    edited["materials"][1]["density"]=2000;
    candidate=pw::parseMaterials(edited.dump());
    bool refused=false;try{pw::validateMaterialReload(library,candidate,live);}catch(const std::exception&){refused=true;}CHECK(refused);
    edited=source;edited["materials"][1]["name"]="Different";
    candidate=pw::parseMaterials(edited.dump());
    refused=false;try{pw::validateMaterialReload(library,candidate,live);}catch(const std::exception&){refused=true;}CHECK(refused);
    std::cout<<"Material validation, reload and portable packing tests passed\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
