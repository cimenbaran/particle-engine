# Particle World

Native C++20 GPU material sandbox based on the five sprint plans in `Tasks/`.
Sprint 1 is in progress. The foundation scene uses gravity and analytic box
contacts; granular pairs, water, reactions and persistent terrain come later.

## Build

Install CMake 3.28+, Ninja, Git, and Xcode on Apple silicon or Visual Studio 2022
with the Windows SDK and C++ tools on Windows. The first build downloads pinned
dependencies and builds bgfx's shader compiler, which can take several minutes.

macOS:

```sh
cmake --preset macos-release
cmake --build --preset macos-release --parallel 8
ctest --preset macos-release
./build/macos-release/src/app/ParticleWorld.app/Contents/MacOS/ParticleWorld
```

Windows, from an x64 Visual Studio developer shell:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release --parallel 8
ctest --preset windows-release
.\build\windows-release\src\app\ParticleWorld.exe
```

The app uses bundle Resources on macOS and assets beside its executable on
Windows, independent of the current working directory. `--frames 180` runs a bounded native smoke test. `--gpu-test` runs the
particle ABI, ordered compute and CPU-reference integration checks on a real graphics device
and exits with a failure code on a mismatch. Startup also runs these checks.
CPU-only tests need neither a graphics device nor downloaded libraries:

```sh
cmake --preset core-tests
cmake --build --preset core-tests
ctest --preset core-tests
```

Package with `cmake --install build/macos-release --prefix out/macos-release`
(or substitute `windows-release`). Packages include precompiled backend shaders;
a shader compiler is needed only when building. Signing/notarization are pending.

## Controls and current limits

Use the ImGui panel to pause, single-step, reset the seeded scene and orbit the
camera. The foundation displays 16,384 particles with a common radius. They
bounce inside analytic bounds and do not yet collide with each other.

The current submission uses a known, initialized instance count. Dynamic GPU
allocation/compaction, frustum classification, the million-particle benchmark,
material loading and shader reload remain Sprint 1 work. See
`docs/Sprint_01_Progress.txt` for verification evidence and open gates.
