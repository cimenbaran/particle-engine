# Particle World

Native C++20 GPU material sandbox based on the five sprint plans in `Tasks/`.
The foundation includes ballistic simulation and an experimental GPU granular
mode with pair contacts and bounded walls. Water, reactions and persistent terrain
remain planned; cross-platform granular stability and release gates are still open.
A resident voxel basin runs with `--terrain-preview`, including interacting sand
in a 1,024-slot command pool and diagnostic collision probes. Add sand, remove,
impulse and clear controls use guarded GPU spawning; see
[basin controls](docs/GPU_Basin_Pool.txt). Live voxel support reports settling candidates.
A [terrain/debris transaction API](docs/GPU_Debris_Transaction.txt) now commits
validated removals, particles and per-material conversion totals together.
Use [Excavate brush](docs/Terrain_Excavation.txt) to turn up to 32 terrain voxels
into physical debris per click, with conversion mass totals shown in the panel.
Explosion planning, deposition and save/load remain open.
Development currently prioritizes macOS.

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
and exits with a failure code on a mismatch. Startup runs the core GPU checks;
`--gpu-test` additionally checks shader reload and physical/cosmetic isolation.
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

## Resident terrain inspection

```sh
open -n ~/particleengine/out/macos-release/ParticleWorld.app --args --terrain-preview
```

Inspect the stone ledge, seeded sand slope, dry pool recess and wood patch. The
panel controls the camera and rebuilds the seed under **Camera and world**.
**Excavate brush** converts up to 32 voxels into debris after GPU validation.
**Add sand** queues a brush in the 1,024-slot interactive pool; **Particle material**
chooses the target for clear, remove and impulse. **Drop interacting diagnostics** starts
256 fixed grains with both particle and terrain collisions. **Drop collision probes**
keeps the terrain-only diagnostic mode. **Pause simulation** stops either mode;
Rebuild clears the particles. See [basin physics](docs/GPU_Basin_Tick.txt) and
[resident basin status](docs/Resident_Basin.txt).

## Controls and current limits

Use the ImGui panel to pause, single-step, reset the seeded scene and orbit the
camera. Select a solid material or the preset's material list, and reload edited material JSON
at a tick boundary. Invalid edits retain the last valid table. See
[material definitions](docs/Materials.md) for the schema and editing workflow.

Use **Lift center particles** to queue an upward velocity change for the next
tick. It affects the selected solid material inside the center box; preset mode
targets the first listed material. Single-step applies it while paused; reset clears pending commands.

The foundation defaults to a 16,384-slot GPU pool with a common particle radius. Use
**Empty scene**, then **Spawn particles**, to add a batch on the next tick.
Set a lifetime in simulation seconds (zero means infinite). **Remove center
particles** frees slots inside the same box used by the lift control. Spawning
and removal target the selected material, or the first preset material. Full pools
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

The local ten-minute Metal soak passed at one million slots. Windows device
validation and other foundation gates remain Sprint 1 work. See
`docs/Sprint_01_Progress.txt` for verification evidence and open gates.

Use `--capabilities out/device.json` to export advertised GPU support and the
selected draw fallback without running the particle scene. See
[capability reporting](docs/Capabilities.txt) for requirements and limitations.

Use **Reload shaders** in the app after rebuilding shader assets. Reload keeps
the scene and paused tick, and retains the working programs on detectable load
failure. See [shader reload](docs/Shader_Pipeline.txt) for the asset workflow and
compatibility limits.

Bulk compute supports 128-thread (default) and 256-thread builds. Configure with
`-DPW_WORKGROUP_SIZE=256` to select the alternate variant; matching shader
assets are required. Both passed local Metal GPU tests. See
[capability reporting](docs/Capabilities.txt) for build and validation details.

Enable **Free camera**, click the scene, and use WASD plus Q/E to move; Shift
boosts speed. Hold right mouse to look, or use **Reset view** to return home.
Camera movement works while paused. See [camera controls](docs/Camera_Controls.txt).

Expand **Pointer brush** to spawn, remove or lift a selected material at a
clicked position. Adjust the horizontal plane and box size; the wire preview
shows the region. Commands apply on the next tick. See
[pointer brushes](docs/Pointer_Brushes.txt).

Use **Scene presets** for a ballistic cloud, empty workspace, or seeded sand
drop. Custom files load with `--scene path.json`; invalid reloads preserve the
current scene. See [scene presets](docs/Scene_Presets.txt) for the versioned schema.

The **Recording** window captures physical commands from a seeded reset. Stop
and save the recording, then replay it in memory or load its file. Playback
starts paused; use single-step or resume. For a command-line replay and GPU
audit, run `--replay capture.json --report replay-result.json`. Recording clears
cosmetics and locks scene/material changes. See [recording controls and file
format](docs/Command_Recordings.txt).

**Cosmetic sparks** uses an independent pool that starts empty. Bursts have
finite lifetimes; overflow drops effects without consuming physical capacity.
Use `--cosmetic-capacity 0` to disable it, or choose up to 150,000 slots. A
benchmark includes this second pool only when explicitly requested, for example
`--capacity 100000 --cosmetic-capacity 150000 --benchmark out/dual --frames 600`.
This filled ballistic workload has no water or pairwise solver. See
[cosmetic controls and limits](docs/Cosmetic_Particles.txt).

The **Allocations** window separates requested buffer payload from renderer
estimates and shows resource handle counts. An unavailable driver counter is
shown explicitly. Benchmark, soak and replay reports include a resource
breakdown; see [memory accounting](docs/Resource_Usage.txt) for exclusions.

Windows CI can run D3D11 correctness checks through an explicitly created WARP
software device using `--software-gpu --gpu-test`. This option is restricted to
tests/capability reporting and does not certify hardware performance. See
[Windows validation](docs/Windows_Validation.txt) for software and device gates.

## Granular mode

Enable **Particle collisions (resets scene)** in the app, or launch with
`--granular` for a 512-slot sand demo (256 initial particles, cosmetics off).
`--capacity` still overrides the capacity. Pause, single-step, reset, solid spawn,
remove and impulse brushes work with the granular solver. The panel shows the
12 m/s limit, substeps, contact counts, speed caps and boundary events. Faults
remain latched on the GPU and pause the UI until reset. The Recording and
Allocations panels start collapsed so the particles remain visible.

Granular capture/replay, shader hot reload and benchmark/soak modes are not yet
connected; ballistic recordings keep their existing behavior. See
[interactive granular controls](docs/Interactive_Granular.txt) and the
[GPU tick coordinator](docs/GPU_Granular_Step.txt) for limits and validation scope.
