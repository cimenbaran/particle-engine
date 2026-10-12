if(APPLE)
  set(PW_SHADER_PLATFORM osx)
  set(PW_SHADER_PROFILE metal)
else()
  set(PW_SHADER_PLATFORM windows)
  set(PW_SHADER_PROFILE s_5_0)
endif()
file(GLOB_RECURSE PW_SHADER_INCLUDES CONFIGURE_DEPENDS ${bgfx_cmake_SOURCE_DIR}/bgfx/src/*.sh)
set(PW_SHADER_OUTPUTS)
foreach(shader vs_particles fs_particles vs_gui fs_gui cs_particles cs_reset cs_export)
  if(shader MATCHES "^vs_")
    set(kind vertex)
  elseif(shader MATCHES "^fs_")
    set(kind fragment)
  else()
    set(kind compute)
  endif()
  if(kind STREQUAL "compute")
    set(folder compute)
  else()
    set(folder render)
  endif()
  set(output ${CMAKE_BINARY_DIR}/shaders/${shader}.bin)
  add_custom_command(OUTPUT ${output} ${output}.json
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND $<TARGET_FILE:shaderc> -f ${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader}.sc
      -o ${output} --type ${kind} --platform ${PW_SHADER_PLATFORM} --profile ${PW_SHADER_PROFILE}
      --varyingdef ${PROJECT_SOURCE_DIR}/shaders/render/varying.def.sc
      -i ${bgfx_cmake_SOURCE_DIR}/bgfx/src
    COMMAND ${CMAKE_COMMAND}
      -DSOURCE=${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader}.sc
      -DVARYING=${PROJECT_SOURCE_DIR}/shaders/render/varying.def.sc
      -DINCLUDE_ROOT=${bgfx_cmake_SOURCE_DIR}/bgfx/src
      -DCOMPILER=$<TARGET_FILE:shaderc> -DBINARY=${output} -DOUTPUT=${output}.json
      -DPLATFORM=${PW_SHADER_PLATFORM} -DPROFILE=${PW_SHADER_PROFILE} -DKIND=${kind}
      -P ${PROJECT_SOURCE_DIR}/cmake/ShaderManifest.cmake
    DEPENDS shaderc ${PW_SHADER_INCLUDES} ${PROJECT_SOURCE_DIR}/cmake/ShaderManifest.cmake ${PROJECT_SOURCE_DIR}/shaders/${folder}/${shader}.sc
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
