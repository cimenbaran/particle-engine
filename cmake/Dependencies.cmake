include(FetchContent)
# Full commit IDs: release builds never resolve moving branches.
set(BGFX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BGFX_INSTALL OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TOOLS_GEOMETRY OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TOOLS_TEXTURE OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TOOLS_BIN2C OFF CACHE BOOL "" FORCE)
# SDL is fetched as a source archive; never describe the parent project Git tree.
set(SDL_REVISION "SDL-3.2.28-7f3ae3d57459e59943a4ecfefc8f6277ec6bf540" CACHE STRING "" FORCE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(bgfx_cmake GIT_REPOSITORY https://github.com/bkaradzic/bgfx.cmake.git
  GIT_TAG a046ef085427ff7dd5a499d7f48e06f5d1bb049d GIT_SUBMODULES bgfx bx bimg)
FetchContent_Declare(SDL3 URL https://github.com/libsdl-org/SDL/archive/7f3ae3d57459e59943a4ecfefc8f6277ec6bf540.tar.gz)
FetchContent_Declare(glm URL https://github.com/g-truc/glm/archive/a532f5b1cf27d6a3c099437e6959cf7e398a0a67.tar.gz)
FetchContent_Declare(json URL https://github.com/nlohmann/json/archive/65ee68451d8eb2b5f3a30b410476ab83deb3289b.tar.gz)
FetchContent_Declare(imgui URL https://github.com/ocornut/imgui/archive/bf75bfec48fc00f532af8926130b70c0e26eb099.tar.gz)
FetchContent_MakeAvailable(bgfx_cmake SDL3 glm json imgui)
if(APPLE)
  include(${CMAKE_CURRENT_LIST_DIR}/PatchBgfxMetal.cmake)
endif()
add_library(pw_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp)
target_include_directories(pw_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(pw_imgui PUBLIC SDL3::SDL3-static)
