#pragma once
#include <cstdint>
#ifndef PW_WORKGROUP_SIZE
#define PW_WORKGROUP_SIZE 128
#endif
namespace pw {
inline constexpr std::uint32_t workgroupSize=PW_WORKGROUP_SIZE;
static_assert(workgroupSize==128 || workgroupSize==256);
// Ceiling division without overflowing at the uint32 limit. A zero workload
// has zero groups; callers omit its dispatch as appropriate for the pass.
constexpr std::uint32_t dispatchGroups(std::uint32_t count) {
    return count/workgroupSize+(count%workgroupSize!=0);
}
}
