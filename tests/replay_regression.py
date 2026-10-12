"""Run a bounded GPU replay with independent pool-count expectations.

Usage: python tests/replay_regression.py PATH_TO_EXECUTABLE OUTPUT_DIRECTORY
       [--software-gpu]
Requires a native graphics session, or explicit Windows D3D11 WARP.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def workload():
    base = Path(__file__).parent / "fixtures/recordings/three-ticks.json"
    recording = json.loads(base.read_text())
    recording.update(capacity=257, ticks=1024, commands=[])
    commands = recording["commands"]

    def add(tick, kind, material, count=0, lifetime=0):
        commands.append(dict(
            tick=tick, sequence=len(commands), type=kind, material=material,
            count=count, minimum=[-8, -1, -8] if kind != "spawn" else [-1, 2, -1],
            maximum=[8, 1024, 8] if kind != "spawn" else [1, 3, 1],
            parameters=[0, 2 if kind == "impulse" else 0, 0, lifetime]))

    for start in range(0, 1024, 128):
        add(start, "spawn", "Stone", 129)
        add(start, "spawn", "Sand", 128)
        add(start, "spawn", "Wood", 16)  # Full pool: exactly 16 rejected.
        add(start + 1, "remove", "Stone")
        add(start + 1, "remove", "Stone")  # Duplicate must not free Sand.
        add(start + 2, "spawn", "Wood", 129, 1)
        add(start + 3, "impulse", "Sand")
        add(start + 70, "remove", "Sand")  # Wood has expired by now.
        add(start + 71, "spawn", "Ash", 257)  # Must fit without rejection.
        add(start + 127, "remove", "Ash")
    add(1023, "spawn", "Wood", 73)  # Nonempty, easily checked final state.
    return recording


def verify(report, recording, software, masked):
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    require(report["passed"] is True, "Replay GPU audit failed")
    require(report["input"] == recording, "Executed input differs from fixture")
    require(report["completed_ticks"] == 1024, "Replay did not complete 1024 ticks")
    require(report["applied_commands"] == 81, "Replay command count differs")
    require(report["capacity"] == 257, "Wrong pool capacity")
    require(report["audit_every_tick"] is True, "Every-tick audit was disabled")
    require(report["final_snapshot"] == dict(
        checked_slots=257, ownership_error=False, state_error=False,
        identity_error=False, active=73, free=184, rejected_spawns=128),
        "Final pool state differs from the independent count oracle")
    require(report["software_renderer_requested"] == software, "Wrong renderer mode")
    require(not masked or report["indirect"] is False, "Masked path was not used")
    if software:
        require(report["backend"] == "Direct3D 11", "WARP requires D3D11")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--software-gpu", action="store_true")
    args = parser.parse_args()
    executable, output = args.executable.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recording = workload()
    fixture = output / "recording.json"
    fixture.write_text(json.dumps(recording, indent=2) + "\n")
    for masked in (False, True):
        name = "masked" if masked else "default"
        report_path = output / f"{name}.json"
        command = [str(executable), "--replay", str(fixture), "--report", str(report_path)]
        if args.software_gpu:
            command.append("--software-gpu")
        if masked:
            command.append("--no-indirect")
        # Executable-relative assets must work outside the source checkout.
        with tempfile.TemporaryDirectory(prefix="particle-replay-") as cwd:
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               timeout=180, check=True)
        report = json.loads(report_path.read_text())
        verify(report, recording, args.software_gpu, masked)
        print(f"PASS {name}: 1024 ticks, 81 commands, active 73/free 184/rejected 128",
              flush=True)


if __name__ == "__main__":
    main()
