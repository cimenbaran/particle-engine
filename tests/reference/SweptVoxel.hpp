#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace pw::reference {
using VoxelVector=std::array<double,3>;
struct VoxelSweepResult {
    bool hit=false,initialOverlap=false;
    double time=1; // Fraction of the supplied start-to-end segment.
    VoxelVector centre{},point{},normal{};
};
inline double voxelDistance(VoxelVector p,VoxelVector lower) {
    VoxelVector d;
    for(unsigned axis=0;axis<3;++axis)d[axis]=p[axis]-std::clamp(p[axis],lower[axis],lower[axis]+0.125);
    return std::hypot(d[0],d[1],d[2]);
}
// Independent double CPU oracle for one closed 0.125m voxel. This returns
// geometry, not a velocity response. Touching at t=0 is a hit even if separating;
// callers must apply their own response/overlap policy. Inputs are origin-relative.
inline VoxelSweepResult sweepVoxelSphere(VoxelVector start,VoxelVector end,VoxelVector lower,double radius=0.0625) {
    if(!std::isfinite(radius) || radius<1e-6 || radius>4)
        throw std::invalid_argument("Invalid voxel sweep radius");
    for(unsigned axis=0;axis<3;++axis) {
        for(double value:{start[axis],end[axis],lower[axis],lower[axis]+0.125})
            if(!std::isfinite(value) || std::abs(value)>1024)
                throw std::invalid_argument("Voxel sweep exceeds bounded reference domain");
    }
    VoxelVector delta;for(unsigned axis=0;axis<3;++axis)delta[axis]=end[axis]-start[axis];
    const auto contact=[&](double time,bool overlap) {
        VoxelSweepResult result;result.hit=true;result.initialOverlap=overlap;result.time=time;
        VoxelVector offset;
        for(unsigned axis=0;axis<3;++axis) {
            result.centre[axis]=std::lerp(start[axis],end[axis],time);
            result.point[axis]=std::clamp(result.centre[axis],lower[axis],lower[axis]+0.125);
            offset[axis]=result.centre[axis]-result.point[axis];
        }
        const double distance=std::hypot(offset[0],offset[1],offset[2]);
        if(distance>0)for(unsigned axis=0;axis<3;++axis)result.normal[axis]=offset[axis]/distance;
        else {
            // An interior centre has no unique projection onto the solid.
            // Choose its nearest boundary, ties in -X,+X,-Y,+Y,-Z,+Z order.
            double closest=1e100;unsigned side=0;
            for(unsigned candidate=0;candidate<6;++candidate) {
                const unsigned axis=candidate/2;
                const double gap=(candidate&1)?lower[axis]+0.125-result.centre[axis]:result.centre[axis]-lower[axis];
                if(gap<closest){closest=gap;side=candidate;}
            }
            result.normal[side/2]=(side&1)?1:-1;
            result.point[side/2]=lower[side/2]+((side&1)?0.125:0);
        }
        return result;
    };
    const double initial=voxelDistance(start,lower);
    if(initial<=radius)return contact(0,initial<radius);
    // Between crossings of the six voxel planes, the closest-point feature is
    // fixed and squared distance is quadratic. This includes rounded edges and
    // corners, unlike intersection with an axis-expanded bounding box.
    std::array<double,8> cuts{0,1};unsigned count=2;
    for(unsigned axis=0;axis<3;++axis)if(delta[axis]!=0)
        for(double plane:{lower[axis],lower[axis]+0.125}) {
            const double time=(plane-start[axis])/delta[axis];
            if(time>0 && time<1)cuts[count++]=time;
        }
    std::sort(cuts.begin(),cuts.begin()+count);
    count=unsigned(std::unique(cuts.begin(),cuts.begin()+count)-cuts.begin());
    for(unsigned i=1;i<count;++i) {
        const double left=cuts[i-1],duration=cuts[i]-left,middle=left+duration*0.5;
        VoxelVector q{},v{};
        for(unsigned axis=0;axis<3;++axis) {
            const double coordinate=std::fma(delta[axis],middle,start[axis]);
            if(coordinate<lower[axis] || coordinate>lower[axis]+0.125) {
                const double plane=coordinate<lower[axis]?lower[axis]:lower[axis]+0.125;
                q[axis]=std::fma(delta[axis],left,start[axis])-plane;v[axis]=delta[axis];
            }
        }
        const double distance=std::hypot(q[0],q[1],q[2]);
        if(distance<=radius)return contact(left,false);
        double a=0,b=0;for(unsigned axis=0;axis<3;++axis){a=std::fma(v[axis],v[axis],a);b=std::fma(q[axis],v[axis],b);}
        if(a==0 || b>=0)continue;
        const double minimumTime=-b/a;
        VoxelVector nearest;for(unsigned axis=0;axis<3;++axis)nearest[axis]=std::fma(v[axis],minimumTime,q[axis]);
        const double closest=std::hypot(nearest[0],nearest[1],nearest[2]);
        if(closest>radius)continue;
        const double discriminant=a*(radius-closest)*(radius+closest);
        const double root=(distance-radius)*(distance+radius)/(-b+std::sqrt(discriminant));
        if(root>=0 && root<=duration) {
            const double time=std::min(1.0,left+root);
            // Subtracting distant endpoints may round away an ULP of the
            // supplied end position. Preserve exact closed-segment semantics.
            if(time==1 && voxelDistance(end,lower)>radius)continue;
            return contact(time,false);
        }
    }
    VoxelSweepResult miss;miss.centre=end;return miss;
}
}
