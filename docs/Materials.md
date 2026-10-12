# Material definitions

`assets/materials/default.json` is the editable source for the seven initial
profiles. Stone, sand, wood and ash can be selected in the current ballistic
scene. Water, smoke and steam have authoring profiles but their fluid, thermal
and gas behavior is not implemented yet. Declared transitions are validated;
they do not execute in this foundation scene.

Build the project after editing the source JSON. The asset builder validates it,
produces `build/macos-release/materials/default.bin`, and copies both files into
the application assets. Asset-only edits trigger this step and refresh the app.
The runtime reads JSON so an installed application remains editable.

For a live edit, change the running app's `assets/materials/default.json`, then
press **Reload materials**. Hover over the button to see the exact path. Reload
commits before the next simulation tick; when paused, use **Single tick**. Bad
JSON or invalid profiles leave the old table running and display an error.
A live material cannot be removed, renamed, change phase or change density;
restart the app to apply these changes and reconstruct particle masses.
**Reset scene** preserves the current loaded table and restores the seeded cloud.

## Schema version 1

The root requires `schema_version: 1`, units `m`, `kg`, `s`, `K`, and 1–256
materials. IDs are unique, contiguous integers starting at zero; array ordering
does not determine the GPU index. Names are unique and 1–64 UTF-8 bytes. Unknown
or missing fields are rejected so spelling mistakes do not silently use defaults.

| Field | Unit | Supported range |
| --- | --- | --- |
| density | kg/m³ | 0.001–100,000 |
| friction | fraction | 0–1 |
| restitution | fraction | 0–1 |
| drag | 1/s | 0–100 |
| conductivity | W/(m K) | 0–10,000 |
| heat_capacity | J/(kg K) | 0.001–1,000,000 |
| ignition_kelvin | K | 0–100,000; zero means non-igniting |
| burn_rate | 1/s | 0–100; positive rates require an ignition threshold |
| color | linear RGBA | four finite values in 0–1 |

`phase` is `solid`, `liquid` or `gas`. `transitions` contains unique target IDs.
Targets must exist and the graph must be acyclic in this schema. These limits
are authoring bounds, not a claim that every future thermal solver is stable
for every accepted combination.

## GPU and packed-file layout

A material is four 16-byte lanes, exactly 64 bytes with 16-byte alignment:

1. Density, friction, restitution, drag (float32).
2. Conductivity, heat capacity, ignition kelvin, burn rate (float32).
3. Red, green, blue, alpha (float32).
4. Stable material ID, phase, render class, reserved zero (uint32).

Phase and render class currently use solid=0, liquid=1, gas=2. Shader kernels
read the table at binding 3. Particle masses derive from sphere volume and
density at reset. Existing steady-state simulation remains on the GPU.

The packed file is little-endian: four ASCII bytes `PWMT`, uint32 version=1,
uint32 stride=64, uint32 count, uint64 schema fingerprint, uint64 content
fingerprint, then the records. Fingerprints are FNV-1a compatibility identifiers,
not cryptographic signatures. The content fingerprint uses canonical JSON
object-key ordering and ignores whitespace; material array order remains part
of the fingerprint. JSON is authoritative; the binary records a build snapshot.

Run the builder independently:

```sh
./build/macos-release/pw_asset_builder assets/materials/default.json /tmp/materials.bin
```

The full native build's CTest suite includes material validation and binary
packing tests. `ParticleWorld --gpu-test` verifies material, particle and render
record layouts on a real device and compares material-driven integration with
a CPU reference. CPU-only `core-tests` intentionally avoids dependencies and
does not run the material JSON tests.
