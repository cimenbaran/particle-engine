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
foreach(shader fs_cosmetics cs_cosmetic_reset cs_cosmetic_spawn cs_cosmetic_counts cs_cosmetic_step vs_voxel_terrain vs_terrain fs_terrain vs_particles fs_particles vs_gui fs_gui cs_particles cs_reset cs_export cs_pool_clear cs_pool_classify cs_pool_spawn cs_pool_spawn_voxel cs_pool_spawn_clear cs_spawn_validate cs_spawn_clearance_basin cs_pool_brush cs_pool_export cs_pool_stats cs_visible_clear cs_visible cs_visible_args cs_indirect_export cs_audit_clear cs_audit_membership cs_audit_state cs_scan_blocks cs_scan_add cs_scan_blocks_saturating cs_scan_add_saturating cs_scan_export cs_radix_flags cs_radix_scatter cs_radix_export cs_key_clear cs_spatial_keys cs_range_clear cs_range_flags cs_range_scatter cs_range_owners cs_range_finalize cs_range_export cs_occupancy_clear cs_occupancy_reduce cs_hash_clear cs_hash_insert cs_hash_finalize cs_hash_lookup cs_hash_export cs_neighbors_test cs_neighbors_export cs_contact_clear cs_contact_totals cs_contact_pack cs_contacts cs_contact_reduce cs_contact_apply cs_velocity_clear cs_velocity_reconstruct cs_velocity_apply cs_response_pack cs_response_degree cs_response_velocity cs_response_reduce cs_response_apply cs_swept_bounds cs_blast_clear cs_blast_propose cs_blast_apply cs_blast_export cs_debris_ack cs_debris_clearance_prepare cs_debris_clearance_pairs_basin cs_debris_clearance_terrain cs_debris_clearance_clear cs_debris_clearance_reduce cs_debris_activate_clear cs_debris_owners cs_debris_claims cs_debris_activate_finalize cs_debris_activate cs_debris_clear cs_debris_flags cs_debris_finalize cs_debris_reserve cs_voxel_ack cs_voxel_sweep cs_voxel_world_sweep cs_voxel_motion cs_voxel_finish cs_voxel_contacts cs_granular_force cs_granular_finish cs_granular_collect cs_granular_latch cs_settling_reset cs_settling_clear cs_settling_support cs_settling_voxels cs_settling_reduce cs_settling_advance cs_particle_debug cs_contact_normals cs_spatial_keys_basin cs_range_flags_basin cs_range_scatter_basin cs_hash_insert_basin cs_hash_lookup_basin cs_neighbors_test_basin cs_contact_pack_basin cs_voxel_contacts_basin cs_contacts_basin cs_response_degree_basin cs_response_velocity_basin cs_settling_support_basin)
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
  set(shader_source ${shader})
  if(shader MATCHES "_saturating$")
    string(REGEX REPLACE "_saturating$" "" shader_source "${shader}")
    string(APPEND shader_defines ";PW_SCAN_SATURATE=1")
  endif()
  if(shader MATCHES "_basin$")
    string(REGEX REPLACE "_basin$" "" shader_source "${shader}")
    string(APPEND shader_defines ";PW_BASIN_GRID=1")
  endif()
  set(output ${CMAKE_BINARY_DIR}/shaders/${shader}.bin)
  add_custom_command(OUTPUT ${output} ${output}.json
    COMMAND ${CMAKE_COMMAND}
      -DSOURCE=${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader_source}.sc
      -DVARYING=${PROJECT_SOURCE_DIR}/shaders/render/varying.def.sc
      -DINCLUDE_ROOT=${bgfx_cmake_SOURCE_DIR}/bgfx/src
      -DPROJECT_INCLUDE_ROOT=${PROJECT_SOURCE_DIR}/shaders
      -DCOMPILER=$<TARGET_FILE:shaderc> -DBINARY=${output} -DOUTPUT=${output}.json
      -DCACHE_DIR=${PW_SHADER_CACHE_DIR} "-DDEFINES=${shader_defines}"
      -DPLATFORM=${PW_SHADER_PLATFORM} -DPROFILE=${PW_SHADER_PROFILE} -DKIND=${kind}
      -P ${PROJECT_SOURCE_DIR}/cmake/CompileShader.cmake
    DEPENDS shaderc ${PROJECT_SOURCE_DIR}/cmake/CompileShader.cmake ${PROJECT_SOURCE_DIR}/cmake/ShaderFingerprint.cmake ${PW_SHADER_INCLUDES} ${PW_PROJECT_SHADER_INCLUDES} ${PROJECT_SOURCE_DIR}/cmake/ShaderManifest.cmake ${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader_source}.sc
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
