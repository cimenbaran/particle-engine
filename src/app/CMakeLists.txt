add_executable(ParticleWorld MACOSX_BUNDLE main.cpp Benchmark.cpp ../render/ShaderReloadTest.cpp ../render/ImGuiRenderer.cpp ../render/Shaders.cpp ../render/GpuLayoutTest.cpp ../render/ParticleAudit.cpp ../render/GpuAuditTest.cpp ../render/ParticleVisibility.cpp ../render/GpuVisibilityTest.cpp ../render/PoolDiagnostics.cpp ../render/ParticlePool.cpp ../render/GpuPoolTest.cpp)
target_link_libraries(ParticleWorld PRIVATE pw_core pw_assets pw_imgui bgfx bx glm::glm nlohmann_json::nlohmann_json)
target_include_directories(ParticleWorld PRIVATE ${bgfx_cmake_SOURCE_DIR}/bgfx/include)
set_target_properties(ParticleWorld PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER com.particleworld.sandbox MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})
include(${PROJECT_SOURCE_DIR}/cmake/Shaders.cmake)
if(APPLE)
  set(PW_ASSET_DEST "ParticleWorld.app/Contents/Resources/assets")
  set(PW_RUNTIME_ASSETS "$<TARGET_BUNDLE_CONTENT_DIR:ParticleWorld>/Resources/assets")
  install(TARGETS ParticleWorld BUNDLE DESTINATION .)
else()
  set(PW_ASSET_DEST "assets")
  set(PW_RUNTIME_ASSETS "$<TARGET_FILE_DIR:ParticleWorld>/assets")
  install(TARGETS ParticleWorld RUNTIME DESTINATION .)
endif()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/assets/ DESTINATION ${PW_ASSET_DEST})
install(DIRECTORY ${CMAKE_BINARY_DIR}/shaders/ DESTINATION ${PW_ASSET_DEST}/shaders)
add_custom_command(TARGET ParticleWorld POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/assets ${PW_RUNTIME_ASSETS}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders ${PW_RUNTIME_ASSETS}/shaders)

# Bundle the exact notices from the dependency revisions used in this build.
set(PW_LICENSE_DIR ${CMAKE_BINARY_DIR}/licenses)
file(MAKE_DIRECTORY ${PW_LICENSE_DIR})
configure_file(${bgfx_cmake_SOURCE_DIR}/LICENSE ${PW_LICENSE_DIR}/bgfx-cmake.txt COPYONLY)
foreach(lib bgfx bx bimg)
  configure_file(${bgfx_cmake_SOURCE_DIR}/${lib}/LICENSE ${PW_LICENSE_DIR}/${lib}.txt COPYONLY)
endforeach()
configure_file(${sdl3_SOURCE_DIR}/LICENSE.txt ${PW_LICENSE_DIR}/SDL3.txt COPYONLY)
configure_file(${imgui_SOURCE_DIR}/LICENSE.txt ${PW_LICENSE_DIR}/ImGui.txt COPYONLY)
configure_file(${glm_SOURCE_DIR}/copying.txt ${PW_LICENSE_DIR}/GLM.txt COPYONLY)
configure_file(${json_SOURCE_DIR}/LICENSE.MIT ${PW_LICENSE_DIR}/json.txt COPYONLY)
add_custom_command(TARGET ParticleWorld POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PW_LICENSE_DIR} ${PW_RUNTIME_ASSETS}/licenses)
install(DIRECTORY ${PW_LICENSE_DIR}/ DESTINATION ${PW_ASSET_DEST}/licenses)
if(MSVC)
  target_compile_options(ParticleWorld PRIVATE /W4 /permissive-)
else()
  target_compile_options(ParticleWorld PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(PW_PACKED_MATERIALS ${CMAKE_BINARY_DIR}/materials/default.bin)
add_custom_command(OUTPUT ${PW_PACKED_MATERIALS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/materials
  COMMAND $<TARGET_FILE:pw_asset_builder> ${PROJECT_SOURCE_DIR}/assets/materials/default.json ${PW_PACKED_MATERIALS}
  DEPENDS pw_asset_builder ${PROJECT_SOURCE_DIR}/assets/materials/default.json VERBATIM)
add_custom_target(pw_materials DEPENDS ${PW_PACKED_MATERIALS})
add_dependencies(ParticleWorld pw_materials)
set_property(TARGET ParticleWorld APPEND PROPERTY LINK_DEPENDS ${PW_PACKED_MATERIALS})
add_custom_command(TARGET ParticleWorld POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PW_PACKED_MATERIALS} ${PW_RUNTIME_ASSETS}/materials/default.bin)
install(FILES ${PW_PACKED_MATERIALS} DESTINATION ${PW_ASSET_DEST}/materials)

set_property(TARGET ParticleWorld APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/cmake/BuildManifest.cmake)
add_custom_command(TARGET ParticleWorld POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:ParticleWorld>
    -DSOURCE_ROOT=${PROJECT_SOURCE_DIR} -DOUTPUT=${PW_RUNTIME_ASSETS}/build.json
    -DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID} -DCOMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}
    -P ${PROJECT_SOURCE_DIR}/cmake/BuildManifest.cmake VERBATIM)
install(FILES ${PW_RUNTIME_ASSETS}/build.json DESTINATION ${PW_ASSET_DEST})

if(BUILD_TESTING)
  add_executable(pw_capability_tests ../../tests/unit/capability_tests.cpp)
  target_link_libraries(pw_capability_tests PRIVATE pw_core nlohmann_json::nlohmann_json)
  target_include_directories(pw_capability_tests PRIVATE ${bgfx_cmake_SOURCE_DIR}/bgfx/include)
  add_test(NAME capability_classification COMMAND pw_capability_tests)
endif()
if(BUILD_TESTING)
  add_test(NAME shader_cache COMMAND ${CMAKE_COMMAND}
    -DROOT=${PROJECT_SOURCE_DIR} -DTEST_DIR=${CMAKE_BINARY_DIR}/shader-cache-test
    -DCOMPILER=$<TARGET_FILE:shaderc> -DINCLUDE_ROOT=${bgfx_cmake_SOURCE_DIR}/bgfx/src
    -DPLATFORM=${PW_SHADER_PLATFORM} -DPROFILE=${PW_SHADER_PROFILE}
    -P ${PROJECT_SOURCE_DIR}/tests/shader_cache_test.cmake)
endif()

if(WIN32)
  target_link_libraries(ParticleWorld PRIVATE bcrypt)
endif()

file(GLOB PW_SCENES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/assets/scenes/*.json)
set_property(TARGET ParticleWorld APPEND PROPERTY LINK_DEPENDS ${PW_SCENES})
