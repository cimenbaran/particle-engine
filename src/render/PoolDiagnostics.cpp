#include "PoolDiagnostics.hpp"
#include "Shaders.hpp"
#include <stdexcept>
namespace pw {
PoolDiagnostics::PoolDiagnostics(const std::filesystem::path& assets) {
    try {
        program_=loadCompute(assets,"cs_pool_stats");
        output_=bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE);
        if(!bgfx::isValid(output_)) throw std::runtime_error("Pool diagnostic image unavailable");
        for(auto& slot:slots_) {
            slot.staging=bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST);
            if(!bgfx::isValid(slot.staging)) throw std::runtime_error("Pool diagnostic staging unavailable");
        }
    }catch(...) {release();throw;}
}
unsigned PoolDiagnostics::pending() const {
    unsigned result=0;for(const auto& slot:slots_) if(slot.busy) ++result;return result;
}
void PoolDiagnostics::release() {
    // Shutdown only: keep destination pointers alive until queued readbacks finish.
    while(pending()) poll(bgfx::frame());
    for(auto& slot:slots_) if(bgfx::isValid(slot.staging)) bgfx::destroy(slot.staging);
    if(bgfx::isValid(output_)) bgfx::destroy(output_);
    if(bgfx::isValid(program_)) bgfx::destroy(program_);
}
PoolDiagnostics::~PoolDiagnostics(){release();}
void PoolDiagnostics::beginEpoch() {++epoch_;latest_={};}
void PoolDiagnostics::poll(std::uint32_t completedFrame) {
    for(auto& slot:slots_) {
        // bgfx frame numbers wrap; every request is at most a few frames old.
        if(!slot.busy || std::int32_t(completedFrame-slot.ready)<0) continue;
        if(slot.epoch==epoch_ && (!latest_.valid || slot.sequence>latest_.sequence))
            latest_={slot.data[0],slot.data[1],slot.data[2],slot.data[3],slot.tick,slot.sequence,true};
        slot.busy=false;
    }
}
bool PoolDiagnostics::request(std::uint16_t view,bgfx::DynamicIndexBufferHandle counts,std::uint64_t tick) {
    for(auto& slot:slots_) if(!slot.busy) {
        bgfx::setViewMode(view,bgfx::ViewMode::Sequential);
        bgfx::setBuffer(0,counts,bgfx::Access::Read);
        bgfx::setImage(1,output_,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);
        bgfx::dispatch(view,program_,1);
        bgfx::TextureRegion dst;dst.handle=slot.staging;
        bgfx::TextureRegion src;src.handle=output_;
        bgfx::blit(std::uint16_t(view+1),dst,src);
        slot.ready=bgfx::read(dst,slot.data.data());
        slot.tick=tick;slot.epoch=epoch_;slot.sequence=sequence_++;slot.busy=true;
        return true;
    }
    return false; // Never wait for or overwrite an in-flight destination.
}
}
