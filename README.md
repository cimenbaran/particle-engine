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
Windows, independent of the current working directory. `--frames 180` runs a bounded native smoke test. `--no-indirect` starts with the masked fallback draw. `--gpu-test` runs the
particle/material/render-record ABI, pool ownership, frustum/compaction,
indirect arguments, ordered compute and
material-driven integration checks on a real graphics device
and exits with a failure code on a mismatch. Startup also runs these checks.
CPU-only tests need neither a graphics device nor downloaded libraries:

```sh
cmake --preset core-tests
cmake --build --preset core-tests
ctest --preset core-tests
```

Package with `cmake --install build/macos-release --prefix out/macos-release`
(or substitute `windows-release`). Packages include precompiled backend shaders;
a shader compiler is needed only when building. Builds reuse a verified shared
[shader cache](docs/Shader_Pipeline.txt). Signing/notarization are pending.

## Controls and current limits

Use the ImGui panel to pause, single-step, reset the seeded scene and orbit the
camera. Select a solid material or mixed solids, and reload edited material JSON
at a tick boundary. Invalid edits retain the last valid table. See
[material definitions](docs/Materials.md) for the schema and editing workflow.

Use **Lift center particles** to queue an upward velocity change for the next
tick. It affects the selected solid material inside the center box; mixed scenes
target Stone. Single-step applies it while paused; reset clears pending commands.

The foundation defaults to a 16,384-slot GPU pool with a common particle radius. Use
**Empty scene**, then **Spawn particles**, to add a batch on the next tick.
Set a lifetime in simulation seconds (zero means infinite). **Remove center
particles** frees slots inside the same box used by the lift control. Spawning
and removal target the selected material, or Stone in a mixed scene. Full pools
reject excess spawns. The panel shows delayed GPU active/free counts and rejected
spawns, labeled with their snapshot tick. CPU command rejections are separate.
Active particles bounce inside analytic bounds and do not yet collide with each other.

GPU frustum culling compacts visible records every frame, including while paused.
Supported devices use a GPU-written indirect draw count. The fallback draws at
fixed capacity and masks unused records. Toggle **Frustum culling** and
**Indirect draw** to compare; the delayed visible count shows culling results.

Capacity is configurable with `--capacity 1000000` (maximum 1,048,576).
For a fixed 1080p benchmark with JSON/CSV output, use
`--benchmark out/benchmarks/local --capacity 1000000 --warmup 120 --frames 600`.
See [benchmark methodology and local results](docs/Benchmarking.txt).

Use `--capacity 1000000 --soak 600 --report out/soak/million.json` for an
audit on every tick and repeated spawn/remove/expiry stress. See
[soak testing](docs/Soak_Testing.txt) for coverage and results.

The ten-minute gate, shader reload, scene/command recording and
Windows device validation remain Sprint 1 work. See
`docs/Sprint_01_Progress.txt` for verification evidence and open gates.

Use `--capabilities out/device.json` to export advertised GPU support and the
selected draw fallback without running the particle scene. See
[capability reporting](docs/Capabilities.txt) for requirements and limitations.
