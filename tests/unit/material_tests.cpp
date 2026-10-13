#include "assets/ExplosionMaterials.hpp"
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
    CHECK(bytes.size()==32+7*72);
    CHECK(bytes[0]==std::byte{'P'} && bytes[1]==std::byte{'W'} && bytes[2]==std::byte{'M'} && bytes[3]==std::byte{'T'});
    CHECK(bytes[4]==std::byte{2} && bytes[8]==std::byte{72} && bytes[12]==std::byte{7});
    const auto expectedBits=std::bit_cast<std::uint32_t>(2600.f);
    for(unsigned i=0;i<4;++i) CHECK(bytes[32+i]==std::byte((expectedBits>>(8*i))&255));
    CHECK(pw::parseMaterials(source.dump(4)).contentHash==library.contentHash);
    auto rejects=[&](auto change){auto candidate=source;change(candidate);bool failed=false;try{pw::parseMaterials(candidate.dump());}catch(const std::exception&){failed=true;}CHECK(failed);};
    rejects([](auto& j){j["schema_version"]=1;});
    rejects([](auto& j){j["units"]["temperature"]="C";});
    rejects([](auto& j){j["materials"][0]["heat_capacity"]=0;});
    rejects([](auto& j){j["materials"][0]["drag"]=-1;});
    rejects([](auto& j){j["materials"][0]["restitution"]=1.01;});
    rejects([](auto& j){j["materials"][1]["id"]=0;});
    rejects([](auto& j){j["materials"][0]["transitions"]={255};});
    rejects([](auto& j){j["materials"][0]["transitions"]={1};j["materials"][1]["transitions"]={0};});
    rejects([](auto& j){j["materials"][0]["color"][0]=2;});
    rejects([](auto& j){j["materials"][0]["typo"]=true;});
    rejects([](auto& j){j["materials"][0].erase("fracture_resistance");});
    for(auto value: {-1.,0.,1e-7,1e12+1}) rejects([&](auto& j){j["materials"][0]["fracture_resistance"]=value;});
    rejects([](auto& j){j["materials"][0]["fracture_resistance"]="strong";});
    rejects([](auto& j){j["materials"][0]["fracture_resistance"]=nullptr;});
    CHECK(library.entries[0].fractureResistance==1000 && library.entries[1].fractureResistance==100);
    CHECK(library.entries[3].fractureResistance==300 && library.entries[4].fractureResistance==50);
    auto precise=source;precise["materials"][0]["fracture_resistance"]=1000.00000001;
    const auto preciseLibrary=pw::parseMaterials(precise.dump());
    CHECK(preciseLibrary.entries[0].fractureResistance==1000.00000001);
    CHECK(preciseLibrary.contentHash!=library.contentHash);
    const auto preciseBytes=pw::encodeMaterials(preciseLibrary);
    const auto resistanceBits=std::bit_cast<std::uint64_t>(1000.00000001);
    for(unsigned i=0;i<8;++i) CHECK(preciseBytes[32+64+i]==std::byte((resistanceBits>>(8*i))&255));
    auto forged=library;forged.entries[0].fractureResistance=1000.00000001;
    CHECK(forged.contentHash==library.contentHash && pw::encodeMaterials(forged)!=bytes);
    for(auto value: {1e-6,1e12}) {auto boundary=source;boundary["materials"][0]["fracture_resistance"]=value;CHECK(pw::parseMaterials(boundary.dump()).entries[0].fractureResistance==value);}
    const auto profiles=pw::explosionMaterials(library);
    CHECK(profiles[0].resistance==1000 && profiles[1].resistance==100 && profiles[3].fuel==1 && !profiles[2].solid);
    auto renamed=source;renamed["materials"][0]["name"]="Custom stone";
    CHECK(pw::explosionMaterials(pw::parseMaterials(renamed.dump()))[0].resistance==1000);
    pw::terrain::SparseVoxelStore store(pw::terrain::ResidentChunkLayout({0,0,0},{1,1,1}),7);
    store.setMaterial(*pw::terrain::addressOf({8,8,8}),0,0);
    pw::terrain::ExplosionRequest blast;blast.centre={1.0625f,1.0625f,1.0625f};blast.radius=.1f;blast.impulse=0;blast.energyJoules=10;
    const auto affordable=pw::terrain::planExplosion(store,blast,0,profiles);
    CHECK(affordable.state==pw::terrain::ExplosionPlan::State::Ready && affordable.removals.size()==1);
    CHECK(affordable.fractureJoules>=1000./512 && affordable.fractureJoules<1000./512+1e-10);
    auto stronger=source;stronger["materials"][0]["fracture_resistance"]=100000;
    const auto rejected=pw::terrain::planExplosion(store,blast,0,pw::explosionMaterials(pw::parseMaterials(stronger.dump())));
    CHECK(rejected.state==pw::terrain::ExplosionPlan::State::InsufficientEnergy && rejected.removals.empty());
    const std::uint32_t live[]{1};
    auto resistanceEdit=source;resistanceEdit["materials"][1]["fracture_resistance"]=200;
    pw::validateMaterialReload(library,pw::parseMaterials(resistanceEdit.dump()),live);
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
