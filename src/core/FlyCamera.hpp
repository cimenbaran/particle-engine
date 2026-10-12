#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace pw {
class FlyCamera {
public:
    FlyCamera() {fromOrbit(.65f,24,12);}
    void fromOrbit(float angle,float distance,float height,float targetHeight=3) {
        position_={std::sin(angle)*distance,height,std::cos(angle)*distance};
        yaw_=-angle;pitch_=std::atan2(targetHeight-height,distance);
    }
    const std::array<float,3>& position() const {return position_;}
    std::array<float,3> forward() const {
        const float horizontal=std::cos(pitch_);
        return {std::sin(yaw_)*horizontal,std::sin(pitch_),-std::cos(yaw_)*horizontal};
    }
    std::array<float,3> target() const {
        auto result=forward();for(unsigned i=0;i<3;++i) result[i]+=position_[i];return result;
    }
    void look(float dx,float dy) {
        if(!std::isfinite(dx) || !std::isfinite(dy)) return;
        yaw_=std::remainder(yaw_+std::clamp(dx,-4096.0f,4096.0f)*.003f,6.283185307f);
        pitch_=std::clamp(pitch_-std::clamp(dy,-4096.0f,4096.0f)*.003f,-1.553343f,1.553343f);
    }
    void move(float ahead,float right,float up,float seconds,float speed) {
        if(!std::isfinite(ahead) || !std::isfinite(right) || !std::isfinite(up) ||
           !std::isfinite(seconds) || !std::isfinite(speed) || seconds<=0 || speed<=0) return;
        const auto direction=forward();
        std::array<float,3> delta{direction[0]*ahead+std::cos(yaw_)*right,
            direction[1]*ahead+up,direction[2]*ahead+std::sin(yaw_)*right};
        const float length=std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        if(!std::isfinite(length) || length<=0) return;
        const float scale=std::min(seconds,.1f)*std::min(speed,100.0f)/std::max(1.0f,length);
        for(unsigned i=0;i<3;++i) position_[i]=std::clamp(position_[i]+delta[i]*scale,i==1?.1f:-128.0f,128.0f);
    }
private:
    std::array<float,3> position_{};
    float yaw_{},pitch_{};
};
}
