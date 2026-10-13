#pragma once
#include "VoxelCoordinates.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace pw::terrain {
// Common-radius granular conversion profile. Quantization is explicit, not
// exact conservation; use one balance per material, never net unlike materials.
inline constexpr float conversionGrainRadius=.0625f;
inline constexpr double maximumConversionError=.05;
struct ConversionMass {
    std::uint32_t voxels{},grains{};
    float inverseGrainMass{};
    double sourceKilograms{},destinationKilograms{},quantizationKilograms{},relativeError{};
    bool admissible=false;
};
struct GrainMass {
    float inverseMass;
    double kilograms,voxelKilograms;
};
inline GrainMass conversionMassProfile(float density) {
    // Same density domain as the material schema. Use the actual encoded
    // inverse mass for accounting; this is the value activation must upload.
    if(!std::isfinite(density) || density<.001f || density>100000.0f)
        throw std::invalid_argument("Invalid conversion material density");
    constexpr double radius=conversionGrainRadius;
    const double nominal=(4.0/3.0)*std::numbers::pi*radius*radius*radius*double(density);
    const float inverse=float(1.0/nominal);
    return {inverse,1.0/double(inverse),voxelMetres*voxelMetres*voxelMetres*double(density)};
}
namespace mass_detail {
inline void tolerance(double value) {
    if(!std::isfinite(value) || value<0 || value>maximumConversionError)
        throw std::invalid_argument("Conversion tolerance must be between zero and five percent");
}
inline std::uint32_t checkedCount(double count) {
    if(count>double(UINT32_MAX))throw std::overflow_error("Conversion product count exceeds uint32");
    return std::uint32_t(count);
}
// Density and inverse mass are binary32: their product fits exactly in a
// binary64 significand. Dividing by 512 is exact. FMA recovers the product
// residual at a half-integer boundary, avoiding an off-by-one at large counts.
inline std::uint32_t debrisCount(std::uint32_t voxels,double factor) {
    const double product=double(voxels)*factor,whole=std::floor(product);
    const double fraction=(product-whole)+std::fma(double(voxels),factor,-product);
    return checkedCount(whole+(fraction>=.5?1:0));
}
inline std::uint32_t depositCount(std::uint32_t grains,double factor) {
    const double whole=std::floor(double(grains)/factor);
    // Compare the midpoint multiplied by the exact factor against the integer
    // source count in one rounding. Exact ties choose the larger product count.
    return checkedCount(whole+(std::fma(whole+.5,factor,-double(grains))<=0?1:0));
}
inline ConversionMass balance(std::uint32_t voxels,std::uint32_t grains,const GrainMass& profile,bool deposit,double tolerance) {
    const double voxelMass=double(voxels)*profile.voxelKilograms,grainMass=double(grains)/double(profile.inverseMass);
    const double source=deposit?grainMass:voxelMass,destination=deposit?voxelMass:grainMass;
    const double residual=std::fma(double(voxels),profile.voxelKilograms*double(profile.inverseMass),-double(grains));
    const double delta=(deposit?residual:-residual)/double(profile.inverseMass);
    const double relative=source>0?std::abs(delta)/source:0;
    return {voxels,grains,profile.inverseMass,source,destination,delta,relative,
            std::abs(delta)<=tolerance*source && (source==0 || destination>0)};
}
}
// Plans only: no terrain edits, slot reservation, mass ledger updates or GPU
// activation. Rejected plans still report their closest integer product count.
inline ConversionMass planVoxelDebris(std::uint32_t voxels,float density,double tolerance=maximumConversionError) {
    mass_detail::tolerance(tolerance);const auto profile=conversionMassProfile(density);
    const auto grains=mass_detail::debrisCount(voxels,double(density)*double(profile.inverseMass)/512);
    return mass_detail::balance(voxels,grains,profile,false,tolerance);
}
// Input grains must all have this material and encoded inverse mass. Arbitrary
// particle masses require a separate measured-mass plan, not a count estimate.
inline ConversionMass planGrainDeposit(std::uint32_t grains,float density,double tolerance=maximumConversionError) {
    mass_detail::tolerance(tolerance);const auto profile=conversionMassProfile(density);
    const auto voxels=mass_detail::depositCount(grains,double(density)*double(profile.inverseMass)/512);
    return mass_detail::balance(voxels,grains,profile,true,tolerance);
}
}
