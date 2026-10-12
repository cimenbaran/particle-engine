#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace pw {
// CPU reference for the bounded grid. Keys belong to cells, never particle IDs.
// UINT32_MAX is reserved for inactive/out-of-domain GPU sort records.
class SpatialGrid {
public:
    using Cell=std::array<std::int32_t,3>;
    using Position=std::array<float,3>;
    static constexpr std::uint32_t invalidKey=UINT32_MAX;
    SpatialGrid(Position origin,float width,Cell minimum,std::array<std::uint32_t,3> dimensions)
        :origin_(origin),width_(width),minimum_(minimum),dimensions_(dimensions) {
        if(!std::isfinite(width) || width<=0) throw std::invalid_argument("Grid width must be finite and positive");
        std::uint64_t count=1;
        for(unsigned axis=0;axis<3;++axis) {
            if(!std::isfinite(origin[axis])) throw std::invalid_argument("Grid origin must be finite");
            if(!dimensions[axis] || count>std::uint64_t(invalidKey)/dimensions[axis])
                throw std::invalid_argument("Grid key space exceeds uint32 capacity");
            const auto last=std::int64_t(minimum[axis])+dimensions[axis]-1;
            if(last>INT32_MAX) throw std::invalid_argument("Grid cell coordinates exceed int32 capacity");
            count*=dimensions[axis];
        }
        cellCount_=std::uint32_t(count);
    }
    std::optional<Cell> coordinates(Position position) const {
        Cell cell;
        for(unsigned axis=0;axis<3;++axis) {
            if(!std::isfinite(position[axis])) return {};
            // Double precision defines the CPU oracle. The GPU implementation
            // must account for rounding at non-power-of-two cell boundaries.
            const double value=std::floor((double(position[axis])-double(origin_[axis]))/double(width_));
            if(value<INT32_MIN || value>INT32_MAX) return {};
            cell[axis]=std::int32_t(value);
        }
        return cell;
    }
    std::optional<std::uint32_t> key(Cell cell) const {
        std::array<std::uint64_t,3> offset;
        for(unsigned axis=0;axis<3;++axis) {
            const auto value=std::int64_t(cell[axis])-minimum_[axis];
            if(value<0 || std::uint64_t(value)>=dimensions_[axis]) return {};
            offset[axis]=std::uint64_t(value);
        }
        return std::uint32_t(offset[0]+std::uint64_t(dimensions_[0])*
            (offset[1]+std::uint64_t(dimensions_[1])*offset[2]));
    }
    std::optional<std::uint32_t> keyAt(Position position) const {
        const auto cell=coordinates(position);return cell?key(*cell):std::nullopt;
    }
    std::optional<Cell> decode(std::uint32_t key) const {
        if(key>=cellCount_) return {};
        Cell cell;
        for(unsigned axis=0;axis<3;++axis) {
            cell[axis]=std::int32_t(std::int64_t(minimum_[axis])+key%dimensions_[axis]);
            key/=dimensions_[axis];
        }
        return cell;
    }
    std::uint32_t searchRadius(float support) const {
        if(!std::isfinite(support) || support<0) throw std::invalid_argument("Grid support must be finite and nonnegative");
        const double radius=std::ceil(double(support)/double(width_));
        if(radius>INT32_MAX) throw std::invalid_argument("Grid search radius exceeds signed coordinate capacity");
        return std::uint32_t(radius);
    }
    std::uint32_t cellCount() const {return cellCount_;}
private:
    Position origin_;
    float width_;
    Cell minimum_;
    std::array<std::uint32_t,3> dimensions_;
    std::uint32_t cellCount_;
};
}
