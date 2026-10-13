#include "GpuGranularStability.hpp"
#include "GpuGranularStep.hpp"
#include "Shaders.hpp"
#include "core/ComputeDispatch.hpp"
#include "core/ParticleTypes.hpp"
#include "assets/Materials.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <utility>
namespace pw {
namespace {
using Json=nlohmann::json;
template<class T> struct Handle {
    T value;explicit Handle(T h):value(h){if(!bgfx::isValid(h))throw std::runtime_error("Stability resource unavailable");}
    ~Handle(){bgfx::destroy(value);}Handle(const Handle&)=delete;
};
struct Scene {
    std::string name;std::vector<ParticleGpu> state;float gravity=9.81f;
    unsigned ticks=600;bool settling=true,closed=false,requiresCap=false;
};
ParticleGpu body(unsigned id,float x,float y,float z) {return {{x,y,z,0.0625f},{0,0,0,1},{293.15f,0,0,0},{0,1,id,1}};}
double energy(const std::vector<ParticleGpu>& state) {
    double sum=0;for(const auto& p:state)for(unsigned a=0;a<3;++a)sum+=0.5*double(p.velocityInvMass[a])*p.velocityInvMass[a]/p.velocityInvMass[3];return sum;
}
struct Metrics {double kinetic=0,potential=0,maxSpeed=0,maxPenetration=0;unsigned contacts=0,deep=0;};
Metrics measure(const std::vector<ParticleGpu>& state,float gravity) {
    Metrics m;m.kinetic=energy(state);for(const auto& p:state)m.potential+=double(gravity)*p.positionRadius[1]/p.velocityInvMass[3];
    for(unsigned i=0;i<state.size();++i){double speed=0;for(unsigned a=0;a<3;++a)speed+=double(state[i].velocityInvMass[a])*state[i].velocityInvMass[a];m.maxSpeed=std::max(m.maxSpeed,std::sqrt(speed));
        for(unsigned j=0;j<i;++j){double squared=0;for(unsigned a=0;a<3;++a){double d=double(state[i].positionRadius[a])-state[j].positionRadius[a];squared+=d*d;}
            const double distance=std::sqrt(squared);if(distance<=0.1251){++m.contacts;const double overlap=std::max(0.0,0.125-distance);m.maxPenetration=std::max(m.maxPenetration,overlap);if(overlap>0.00625)++m.deep;}}
    }return m;
}
// Read-only forensic data from an already verified sample. This adds no GPU
// submissions and does not participate in any gate or simulation decision.
Json inspectSample(const std::vector<ParticleGpu>& state) {
    Json particles=Json::array();unsigned fastest=0;double maximum=-1;
    for(unsigned i=0;i<state.size();++i){const auto& p=state[i];double squared=0;
        for(unsigned a=0;a<3;++a)squared+=double(p.velocityInvMass[a])*p.velocityInvMass[a];
        if(squared>maximum){maximum=squared;fastest=i;}
        particles.push_back({{"slot",p.identity[2]},{"generation",p.identity[3]},
            {"position",{p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]}},
            {"velocity",{p.velocityInvMass[0],p.velocityInvMass[1],p.velocityInvMass[2]}},
            {"radius",p.positionRadius[3]},{"inverse_mass",p.velocityInvMass[3]}});
    }
    if(state.empty())return {{"particle_state",particles},{"fastest_particle",nullptr}};
    const auto& query=state[fastest];Json near=Json::array(),walls=Json::array();
    for(unsigned axis=0;axis<3;++axis){
        walls.push_back(double(query.positionRadius[axis])-(axis==1?0.0:-8.0)-query.positionRadius[3]);
        walls.push_back((axis==1?16.0:8.0)-double(query.positionRadius[axis])-query.positionRadius[3]);
    }
    for(unsigned i=0;i<state.size();++i){if(i==fastest)continue;
        std::array<double,3> delta{},relative{};double squared=0;
        for(unsigned a=0;a<3;++a){delta[a]=double(query.positionRadius[a])-state[i].positionRadius[a];squared+=delta[a]*delta[a];relative[a]=double(query.velocityInvMass[a])-state[i].velocityInvMass[a];}
        const double distance=std::sqrt(squared),gap=distance-query.positionRadius[3]-state[i].positionRadius[3];
        if(gap>0.002)continue;
        Json contact={{"slot",state[i].identity[2]},{"surface_gap",gap},{"normal",nullptr},{"relative_normal_speed",nullptr},{"relative_tangent_speed",nullptr}};
        if(distance>0){
            for(double& d:delta)d/=distance;
            double normalSpeed=0;for(unsigned a=0;a<3;++a)normalSpeed+=relative[a]*delta[a];
            double tangentSquared=0;for(unsigned a=0;a<3;++a){const double v=relative[a]-normalSpeed*delta[a];tangentSquared+=v*v;}
            contact["normal"]=delta;contact["relative_normal_speed"]=normalSpeed;contact["relative_tangent_speed"]=std::sqrt(tangentSquared);
        }
        near.push_back(std::move(contact));
    }
    return {{"particle_state",std::move(particles)},{"fastest_particle",{{"slot",query.identity[2]},
        {"speed",std::sqrt(maximum)},{"wall_gaps",std::move(walls)},{"nearby_particles",std::move(near)}}}};
}
Json sceneCheck(const std::filesystem::path& assets,const Scene& scene) {
    const auto started=std::chrono::steady_clock::now();const auto count=std::uint32_t(scene.state.size());
    Json result={{"name",scene.name},{"capacity",count},{"requested_ticks",scene.ticks},{"gravity",scene.gravity},{"settling_required",scene.settling},{"closed_energy_required",scene.closed},{"cap_required",scene.requiresCap},{"samples",Json::array()},{"passed",false},{"failure",""}};
    result["initial_state"]=Json::array();
    for(const auto& p:scene.state)result["initial_state"].push_back({{"position",{p.positionRadius[0],p.positionRadius[1],p.positionRadius[2]}},{"velocity",{p.velocityInvMass[0],p.velocityInvMass[1],p.velocityInvMass[2]}}});
    bgfx::VertexLayout layout;layout.begin().add(bgfx::Attrib::TexCoord0,4,bgfx::AttribType::Float).end();
    MaterialGpu material{};material.mechanical[0]=1;material.mechanical[1]=0.5f;material.mechanical[2]=0;
    Handle materials(bgfx::createDynamicVertexBuffer(bgfx::copy(&material,sizeof(material)),layout,BGFX_BUFFER_COMPUTE_READ));
    Handle input(bgfx::createDynamicVertexBuffer(bgfx::copy(scene.state.data(),count*sizeof(ParticleGpu)),layout,BGFX_BUFFER_COMPUTE_READ));
    Handle exporter(loadCompute(assets,"cs_export")),statExporter(loadCompute(assets,"cs_pool_stats"));
    Handle uniform(bgfx::createUniform("u_sim",bgfx::UniformType::Vec4));
    Handle image(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
    Handle staging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
    Handle oldImage(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
    Handle oldStaging(bgfx::createTexture2D(4,std::uint16_t(count),false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
    Handle statImage(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_COMPUTE_WRITE));
    Handle statStaging(bgfx::createTexture2D(1,1,false,1,bgfx::TextureFormat::RGBA32U,BGFX_TEXTURE_READ_BACK|BGFX_TEXTURE_BLIT_DST));
    GpuGranularStep solver(assets,count,{},4,scene.gravity,true);auto current=input.value;
    std::vector<ParticleGpu> state(count),original(count);std::array<std::uint32_t,4> status{};
    const double initialEnergy=energy(scene.state);double tailEnergy=initialEnergy,maxEnergy=initialEnergy;bool sawCap=false;
    Metrics final;unsigned tick=0;float expectedAge=0;double tailMaxSpeed=0;bool tailOverlapPassed=true;
    try {
        for(tick=1;tick<=scene.ticks;++tick){
            solver.submit(0,current,materials.value,1);current=solver.output();
            for(unsigned substep=0;substep<solver.plan().substeps;++substep)expectedAge+=solver.plan().substepSeconds;
            if(tick!=1 && tick%30!=0 && tick!=scene.ticks){bgfx::frame();SDL_PumpEvents();continue;}
            bgfx::setViewMode(1,bgfx::ViewMode::Sequential);
            const float params[]{0,float(count),0,4};
            for(auto pair:{std::pair{current,image.value},std::pair{input.value,oldImage.value}}){
                bgfx::setUniform(uniform.value,params);bgfx::setBuffer(0,pair.first,bgfx::Access::Read);bgfx::setImage(1,pair.second,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,exporter.value,dispatchGroups(count));
            }
            bgfx::setBuffer(0,solver.status(),bgfx::Access::Read);bgfx::setImage(1,statImage.value,0,bgfx::Access::Write,bgfx::TextureFormat::RGBA32U);bgfx::dispatch(1,statExporter.value,1);
            bgfx::TextureRegion src,dst;src.handle=image.value;dst.handle=staging.value;bgfx::blit(2,dst,src);auto ready=bgfx::read(dst,state.data());
            src.handle=oldImage.value;dst.handle=oldStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,original.data()));
            src.handle=statImage.value;dst.handle=statStaging.value;bgfx::blit(2,dst,src);ready=std::max(ready,bgfx::read(dst,status.data()));
            while(bgfx::frame()<ready)SDL_PumpEvents();
            if(status[0])throw std::runtime_error("Latched solver phase error "+std::to_string(status[0]));
            if(std::memcmp(original.data(),scene.state.data(),count*sizeof(ParticleGpu)))throw std::runtime_error("Original fixture mutated");
            for(unsigned i=0;i<count;++i){const auto& p=state[i];
                if(std::memcmp(p.identity,scene.state[i].identity,16) || p.positionRadius[3]!=0.0625f || p.velocityInvMass[3]!=1 ||
                   p.thermalState[0]!=293.15f || p.thermalState[1]!=0 || p.thermalState[2]!=0)throw std::runtime_error("Persistent identity/material/mass changed");
                for(unsigned a=0;a<4;++a)if(!std::isfinite(p.positionRadius[a]) || !std::isfinite(p.velocityInvMass[a]) || !std::isfinite(p.thermalState[a]))throw std::runtime_error("Nonfinite state");
                for(unsigned a=0;a<3;++a)if(p.positionRadius[a]<(a==1?0.0625f:-7.9375f) || p.positionRadius[a]>(a==1?15.9375f:7.9375f))throw std::runtime_error("Sphere escaped enclosure");
                if(std::abs(double(p.thermalState[3])-expectedAge)>1e-6)throw std::runtime_error("Lifetime clock differs from float substeps");
            }
            final=measure(state,scene.gravity);maxEnergy=std::max(maxEnergy,final.kinetic);sawCap|=status[1]>0;
            result["samples"].push_back({{"tick",tick},{"age_error_seconds",double(state[0].thermalState[3])-tick/60.0},{"energy",final.kinetic},{"potential_energy",final.potential},{"mechanical_energy",final.kinetic+final.potential},{"max_speed",final.maxSpeed},{"contacts",final.contacts},{"deep_contacts",final.deep},{"max_penetration",final.maxPenetration},{"solver_caps",status[1]},{"solver_max_degree",status[2]},{"boundary_events",status[3]}});
            result["samples"].back().update(inspectSample(state));
            if(final.maxSpeed>12.0001)throw std::runtime_error("Speed limit exceeded");
            if(tick==scene.ticks/2)tailEnergy=final.kinetic;
            if(tick>=scene.ticks-std::min(120u,scene.ticks)) {tailMaxSpeed=std::max(tailMaxSpeed,final.maxSpeed);tailOverlapPassed&=std::uint64_t(final.deep)*100<=final.contacts;}
        }
        if(scene.settling && (!tailOverlapPassed || tailMaxSpeed>0.1))throw std::runtime_error("Settling window: >1% deep contacts or speed >0.1 m/s");
        if(scene.closed && (maxEnergy>initialEnergy+1e-4*std::max(1.0,initialEnergy) || final.kinetic>tailEnergy+1e-4*std::max(1.0,initialEnergy)))throw std::runtime_error("Closed dissipative energy grew");
        if(scene.requiresCap && !sawCap)throw std::runtime_error("High-speed fixture did not report a cap");
        result["passed"]=true;
    }catch(const std::exception& error){result["failure"]=error.what();}
    result["submitted_ticks"]=std::min(tick,scene.ticks);result["last_verified_tick"]=result["samples"].empty()?0:result["samples"].back()["tick"].get<unsigned>();result["tail_max_speed"]=tailMaxSpeed;result["tail_overlap_passed"]=tailOverlapPassed;result["initial_mechanical_energy"]=measure(scene.state,scene.gravity).kinetic+measure(scene.state,scene.gravity).potential;result["last_solver_status"]=status;result["initial_energy"]=initialEnergy;result["maximum_sampled_energy"]=maxEnergy;
    result["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout<<scene.name<<": "<<(result["passed"].get<bool>()?"PASS":"FAIL")<<"; max penetration "<<final.maxPenetration<<"; deep "<<final.deep<<"/"<<final.contacts<<"; speed "<<final.maxSpeed<<"; "<<result["failure"].get<std::string>()<<std::endl;
    return result;
}
}
nlohmann::json runGpuGranularStability(const std::filesystem::path& assets) {
    std::vector<Scene> scenes;
    Scene column;column.name="falling_column_16";for(unsigned i=0;i<16;++i)column.state.push_back(body(i,0,2.0625f+i*0.125f,0));scenes.push_back(column);
    Scene pile;pile.name="resting_pile_64";for(unsigned y=0;y<4;++y)for(unsigned z=0;z<4;++z)for(unsigned x=0;x<4;++x)pile.state.push_back(body(unsigned(pile.state.size()),x*0.125f,0.0625f+y*0.125f,z*0.125f));scenes.push_back(pile);
    Scene wall=pile;wall.name="wall_pile_64";for(auto& p:wall.state){p.positionRadius[0]-=7.9375f;p.positionRadius[1]+=1;p.velocityInvMass[0]=-2;}scenes.push_back(wall);
    Scene dense;dense.name="coincident_spawn_64";dense.ticks=600;for(unsigned i=0;i<64;++i)dense.state.push_back(body(i,0,1,0));scenes.push_back(dense);
    Scene closed=pile;closed.name="closed_dissipative_64";closed.gravity=0;closed.settling=false;closed.closed=true;
    std::mt19937 rng(20261013);auto velocity=[&]{return 4.0f*float(rng()>>8)/16777216.0f-2.0f;};
    for(auto& p:closed.state){p.positionRadius[0]*=1.2f;p.positionRadius[1]=4+(p.positionRadius[1]-0.0625f)*1.2f;p.positionRadius[2]*=1.2f;for(unsigned a=0;a<3;++a)p.velocityInvMass[a]=velocity();}scenes.push_back(closed);
    Scene shots;shots.name="high_speed_wall_129";shots.ticks=30;shots.settling=false;shots.requiresCap=true;shots.gravity=0;
    for(unsigned i=0;i<129;++i){auto p=body(i,7.8f,1+float(i/8)*0.15f,-1+float(i%8)*0.15f);p.velocityInvMass[0]=1000;shots.state.push_back(p);}scenes.push_back(shots);
    Scene grazing;grazing.name="grazing_floor";grazing.gravity=0;grazing.settling=false;grazing.closed=true;grazing.ticks=60;auto p=body(0,0,0.0625f,0);p.velocityInvMass[0]=3;grazing.state.push_back(p);scenes.push_back(grazing);
    Json report={{"schema_version",1},{"workload","granular_stability_v1"},{"position_friction","static_traction_floor_v2"},{"backend",bgfx::getRendererName(bgfx::getCaps()->rendererType)},{"workgroup_size",workgroupSize},{"tick_seconds",1.0/60},{"substeps",4},{"iterations",4},{"speed_limit",12},{"radius",0.0625},{"restitution",0},{"friction",0.5},{"passed",true},{"scenes",Json::array()}};
    for(const auto& scene:scenes){
        Json result;
        try {result=sceneCheck(assets,scene);}
        catch(const std::exception& error){result={{"name",scene.name},{"passed",false},{"failure",error.what()},{"submitted_ticks",0},{"last_verified_tick",0}};}
        if(!result["passed"].get<bool>())report["passed"]=false;report["scenes"].push_back(std::move(result));
    }
    const auto* caps=bgfx::getCaps();report["machine"]={{"platform",SDL_GetPlatform()},{"ram_mb",SDL_GetSystemRAM()},{"vendor_id",caps->vendorId},{"device_id",caps->deviceId}};
    report["notes"]={"Snapshots at tick 1 and every 30 ticks; solver errors latch every tick.","Settling over the final 120 ticks: sampled contact overlap <=10% radius for at least 99% of contacts, maximum speed <=0.1 m/s.","Closed energy is sampled, with 1e-4 relative-to-initial tolerance; not an every-tick energy proof.","Fixed short fixtures, not a ten-minute soak, general stability proof or performance benchmark.","Each verified sample records all particle positions/velocities and the fastest particle nearby-contact geometry; these fields do not change gates.","Fastest-particle wall gaps are x-min, x-max, y-min, y-max, z-min, z-max; nearby normals point toward that particle, and negative normal speed means approach."};
    return report;
}
}
