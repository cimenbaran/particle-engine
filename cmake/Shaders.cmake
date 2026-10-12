if(APPLE)
  set(PW_SHADER_PLATFORM osx)
  set(PW_SHADER_PROFILE metal)
else()
  set(PW_SHADER_PLATFORM windows)
  set(PW_SHADER_PROFILE s_5_0)
endif()
if(WIN32 AND DEFINED ENV{LOCALAPPDATA})
  set(default_shader_cache "$ENV{LOCALAPPDATA}/ParticleWorld/shaders")
else()
  set(default_shader_cache "$ENV{HOME}/.cache/particleworld/shaders")
endif()
set(PW_SHADER_CACHE_DIR "${default_shader_cache}" CACHE PATH "Shared content-addressed shader cache")
file(GLOB_RECURSE PW_SHADER_INCLUDES CONFIGURE_DEPENDS ${bgfx_cmake_SOURCE_DIR}/bgfx/src/*.sh)
file(GLOB_RECURSE PW_PROJECT_SHADER_INCLUDES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/shaders/*.sh)
set(PW_SHADER_OUTPUTS)
foreach(shader vs_particles fs_particles vs_gui fs_gui cs_particles cs_reset cs_export cs_pool_clear cs_pool_classify cs_pool_spawn cs_pool_brush cs_pool_export cs_pool_stats cs_visible_clear cs_visible cs_visible_args cs_indirect_export cs_audit_clear cs_audit_membership cs_audit_state cs_scan_blocks cs_scan_add cs_scan_export cs_radix_flags cs_radix_scatter cs_radix_export cs_key_clear cs_spatial_keys cs_range_clear cs_range_flags cs_range_scatter cs_range_owners cs_range_finalize cs_range_export cs_hash_clear cs_hash_insert cs_hash_finalize cs_hash_lookup cs_hash_export cs_neighbors_test cs_neighbors_export)
  if(shader MATCHES "^vs_")
    set(kind vertex)
  elseif(shader MATCHES "^fs_")
    set(kind fragment)
  else()
    set(kind compute)
  endif()
  if(kind STREQUAL "compute")
    set(folder compute)
    set(shader_defines "PW_WORKGROUP_SIZE=${PW_WORKGROUP_SIZE}")
  else()
    set(folder render)
    set(shader_defines "")
  endif()
  set(output ${CMAKE_BINARY_DIR}/shaders/${shader}.bin)
  add_custom_command(OUTPUT ${output} ${output}.json
    COMMAND ${CMAKE_COMMAND}
      -DSOURCE=${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader}.sc
      -DVARYING=${PROJECT_SOURCE_DIR}/shaders/render/varying.def.sc
      -DINCLUDE_ROOT=${bgfx_cmake_SOURCE_DIR}/bgfx/src
      -DPROJECT_INCLUDE_ROOT=${PROJECT_SOURCE_DIR}/shaders
      -DCOMPILER=$<TARGET_FILE:shaderc> -DBINARY=${output} -DOUTPUT=${output}.json
      -DCACHE_DIR=${PW_SHADER_CACHE_DIR} -DDEFINES=${shader_defines}
      -DPLATFORM=${PW_SHADER_PLATFORM} -DPROFILE=${PW_SHADER_PROFILE} -DKIND=${kind}
      -P ${PROJECT_SOURCE_DIR}/cmake/CompileShader.cmake
    DEPENDS shaderc ${PROJECT_SOURCE_DIR}/cmake/CompileShader.cmake ${PROJECT_SOURCE_DIR}/cmake/ShaderFingerprint.cmake ${PW_SHADER_INCLUDES} ${PW_PROJECT_SHADER_INCLUDES} ${PROJECT_SOURCE_DIR}/cmake/ShaderManifest.cmake ${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader}.sc
      ${PROJECT_SOURCE_DIR}/shaders/render/varying.def.sc
      ${bgfx_cmake_SOURCE_DIR}/bgfx/src/bgfx_shader.sh
      ${bgfx_cmake_SOURCE_DIR}/bgfx/src/bgfx_compute.sh
    VERBATIM)
  list(APPEND PW_SHADER_OUTPUTS ${output} ${output}.json)
endforeach()
add_custom_target(pw_shaders DEPENDS ${PW_SHADER_OUTPUTS})
add_dependencies(ParticleWorld pw_shaders)
# Shader-only edits must also refresh the app's packaged assets.
set_property(TARGET ParticleWorld APPEND PROPERTY LINK_DEPENDS ${PW_SHADER_OUTPUTS})
