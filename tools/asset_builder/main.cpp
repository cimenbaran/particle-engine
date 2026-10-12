#include "assets/Materials.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=3) {std::cerr<<"Usage: pw_asset_builder materials.json materials.bin\n";return 2;}
        const auto library=pw::loadMaterials(argv[1]);
        const auto bytes=pw::encodeMaterials(library);
        std::ofstream output(argv[2],std::ios::binary|std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
        output.close();
        if(!output) throw std::runtime_error("Cannot write packed material output");
        std::cout<<"Packed "<<library.entries.size()<<" materials; schema "<<std::hex<<pw::fingerprint(pw::materialSchema)<<"; content "<<library.contentHash<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Material build: "<<e.what()<<'\n';return 1;}
}
