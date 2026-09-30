"""Capture Mixxx 2.5.4's normal deck analysis in fresh isolated settings.

Database wire formats are defined by upstream src/proto/beats.proto and
src/track/beats.cpp at the revision pinned in mixxx-benchmark.v1.json. No Mixxx
analysis algorithm is reimplemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sqlite3
import struct
import subprocess
import time


KEY_TONICS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def protobuf_fields(data: bytes) -> list[tuple[int, int, object]]:
    """Read only protobuf wire types used by the pinned beat schema, strictly."""
    position = 0

    def varint() -> int:
        nonlocal position
        value = 0
        for shift in range(0, 70, 7):
            if position >= len(data):
                raise RuntimeError("truncated Mixxx protobuf varint")
            byte = data[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                if value >= 2 ** 64:
                    raise RuntimeError("overflowing Mixxx protobuf varint")
                return value
        raise RuntimeError("overflowing Mixxx protobuf varint")

    fields = []
    while position < len(data):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if not field:
            raise RuntimeError("invalid Mixxx protobuf field")
        if wire == 0:
            value = varint()
        else:
            if wire == 1:
                length = 8
            elif wire == 2:
                length = varint()
            elif wire == 5:
                length = 4
            else:
                raise RuntimeError("unsupported Mixxx protobuf wire type")
            if position + length > len(data):
                raise RuntimeError("truncated Mixxx protobuf field")
            value = data[position:position + length]
            position += length
        fields.append((field, wire, value))
    return fields


def unique_field(fields: list, number: int, wire: int, default=None):
    matches = [(kind, value) for field, kind, value in fields if field == number]
    if not matches:
        return default
    if len(matches) != 1 or matches[0][0] != wire:
        raise RuntimeError("invalid or duplicate Mixxx protobuf field")
    return matches[0][1]


def beat_frame(data: bytes) -> int:
    fields = protobuf_fields(data)
    position = unique_field(fields, 1, 0)
    if position is None or unique_field(fields, 2, 0, 1) != 1:
        raise RuntimeError("missing or disabled Mixxx beat")
    if unique_field(fields, 3, 0, 0) != 0:
        raise RuntimeError("Mixxx beat came from metadata/user editing")
    # proto2 int32 uses a 64-bit varint for negative values.
    position &= 0xffffffff
    return position - 2 ** 32 if position >= 2 ** 31 else position


def decode_beats(blob: bytes, version: str, sample_rate: int, duration: float) -> list[float]:
    if sample_rate <= 0 or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("invalid Mixxx sample rate/duration")
    fields = protobuf_fields(blob)
    if version == "BeatGrid-2.0":
        bpm_fields = protobuf_fields(unique_field(fields, 1, 2, b""))
        raw = unique_field(bpm_fields, 1, 1)
        if raw is None or unique_field(bpm_fields, 2, 0, 0) != 0:
            raise RuntimeError("missing or imported Mixxx BPM")
        bpm = struct.unpack("<d", raw)[0]
        if not math.isfinite(bpm) or not 1 <= bpm <= 1000:
            raise RuntimeError("invalid Mixxx BPM")
        first = beat_frame(unique_field(fields, 2, 2, b"")) / sample_rate
        period = 60 / bpm
        start = math.ceil(-first / period)
        stop = math.ceil((duration - first) / period)
        return [first + index * period for index in range(start, stop)]
    if version == "BeatMap-1.0":
        frames = []
        for field, wire, value in fields:
            if field == 1:
                if wire != 2:
                    raise RuntimeError("invalid Mixxx beat-map field")
                frames.append(beat_frame(value))
        if len(frames) < 2 or any(b <= a for a, b in zip(frames, frames[1:])):
            raise RuntimeError("non-increasing or incomplete Mixxx beat map")
        return [frame / sample_rate for frame in frames if 0 <= frame < duration * sample_rate]
    raise RuntimeError(f"unsupported Mixxx beats version: {version}")


def read_database(path: pathlib.Path, expected_paths: list[str]) -> list[dict]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT t.location AS path, l.bpm, l.duration, l.samplerate,
                   l.key_id, l.beats, l.beats_version, l.beats_sub_version,
                   l.keys_version, l.keys_sub_version
            FROM library l JOIN track_locations t ON l.location = t.id
        """).fetchall()
    by_path = {row["path"]: row for row in rows}
    if set(by_path) != set(expected_paths):
        raise RuntimeError("Mixxx database does not contain exactly the requested tracks")
    results = []
    for path in expected_paths:
        row = by_path[path]
        if row["keys_version"] != "KeyMap-1.0" or row["keys_sub_version"] != "vamp_plugin_id=qm-keydetector:2":
            raise RuntimeError("Mixxx did not persist full-track Queen Mary key analysis")
        if row["beats"]:
            if row["beats_sub_version"] != "rounding=V4|vamp_plugin_id=qm-tempotracker:0":
                raise RuntimeError("Mixxx did not persist full-track Queen Mary beat analysis")
            beats = decode_beats(row["beats"], row["beats_version"], row["samplerate"], row["duration"])
        else:
            beats = []
        key_id = row["key_id"]
        if not isinstance(key_id, int) or not 0 <= key_id <= 24:
            raise RuntimeError("invalid Mixxx chromatic key ID")
        key = None if key_id == 0 else KEY_TONICS[(key_id - 1) % 12] + (" major" if key_id <= 12 else " minor")
        bpm = row["bpm"]
        if bpm is not None and (not math.isfinite(bpm) or bpm < 0):
            raise RuntimeError("invalid Mixxx database BPM")
        results.append({
            "path": path, "bpm": bpm or None, "beats": beats, "key": key,
            "durationSeconds": row["duration"], "sampleRate": row["samplerate"],
            "beatsVersion": row["beats_version"], "beatsSubVersion": row["beats_sub_version"],
            "keysSubVersion": row["keys_sub_version"],
        })
    return results


def verify_settings(path: pathlib.Path, fixed_tempo: bool) -> None:
    values = {}
    section = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
        elif line and section:
            key, _, value = line.partition(" ")
            values[(section, key)] = value.strip()
    expected = {
        ("[BPM]", "BeatDetectionFixedTempoAssumption"): str(int(fixed_tempo)),
        ("[BPM]", "BPMDetectionEnabled"): "1",
        ("[BPM]", "FastAnalysisEnabled"): "0",
        ("[Key]", "KeyDetectionEnabled"): "1",
        ("[Key]", "FastAnalysisEnabled"): "0",
        ("[Vamp]", "AnalyserBeatPluginID"): "qm-tempotracker:0",
        ("[Vamp]", "AnalyserKeyPluginID"): "qm-keydetector:2",
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Mixxx persisted settings differ from the requested profile")


def ready_to_flush(output: str, track_count: int) -> bool:
    # In the pinned AnalyzerThread, this is emitted after all PCM processing;
    # keys/silence then finalize in the same uninterrupted finish loop. GUI quit
    # joins those workers. A constant/unknown key need not print a histogram.
    # Persisted plugin provenance is checked after the clean shutdown.
    return output.count("AnalyzerBeats plugin detected") == track_count


def run_batch(paths: list[str], fixed_tempo: bool, directory: pathlib.Path, env: dict) -> list[dict]:
    directory.mkdir()
    config = f"""[Config]
Version 2.5.4
hide_menubar 0
[Library]
RescanOnStartup 0
[BPM]
BPMDetectionEnabled 1
BeatDetectionFixedTempoAssumption {int(fixed_tempo)}
FastAnalysisEnabled 0
ReanalyzeImported 1
ReanalyzeWhenSettingsChange 1
[Key]
KeyDetectionEnabled 1
FastAnalysisEnabled 0
ReanalyzeWhenSettingsChange 1
[Vamp]
AnalyserBeatPluginID qm-tempotracker:0
AnalyserKeyPluginID qm-keydetector:2
"""
    (directory / "mixxx.cfg").write_text(config)
    log_path = directory / "process.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([
            "mixxx", "--settings-path", str(directory), "--safe-mode", "--locale", "en_US",
            "--log-level", "debug", "--log-flush-level", "debug", *paths,
        ], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Mixxx exited before capture; see {log_path}")
                windows = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", "."], env=env, capture_output=True, text=True)
                main = None
                modal = False
                for window in windows.stdout.splitlines():
                    title = subprocess.run(["xdotool", "getwindowname", window], env=env, capture_output=True, text=True).stdout.strip()
                    if title == "Mixxx":
                        main = window
                    elif title in {"Choose music library directory", "No Output Devices", "Allow Mixxx to hide the menu bar?"}:
                        modal = True
                        subprocess.run(["xdotool", "windowfocus", "--sync", window], env=env, check=True)
                        subprocess.run(["xdotool", "key", "Escape" if title == "Choose music library directory" else "Return"], env=env, check=True)
                output = log_path.read_text(errors="replace")
                if main and not modal and ready_to_flush(output, len(paths)):
                    # All PCM was processed. Graceful GUI quit joins workers
                    # and flushes dirty tracks; forced termination is never evidence.
                    subprocess.run(["xdotool", "windowfocus", "--sync", main], env=env, check=True)
                    subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+q"], env=env, check=True)
                    if process.wait(timeout=30) != 0:
                        raise RuntimeError(f"Mixxx failed to shut down cleanly; see {log_path}")
                    verify_settings(directory / "mixxx.cfg", fixed_tempo)
                    return read_database(directory / "mixxxdb.sqlite", paths)
                time.sleep(0.25)
            raise RuntimeError(f"Mixxx analysis timed out; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def capture_job(job_path: pathlib.Path) -> None:
    job = json.loads(job_path.read_text())
    def verify_inputs() -> None:
        for case in job["cases"]:
            actual = hashlib.sha256(pathlib.Path(case["containerPath"]).read_bytes()).hexdigest()
            if actual != case["sha256"]:
                raise RuntimeError(f"Mixxx input changed: {case['name']}")

    verify_inputs()
    package_version = subprocess.check_output(["dpkg-query", "-W", "-f=${Version}", "mixxx"], text=True)
    if package_version != job["config"]["mixxxPackageVersion"]:
        raise RuntimeError("Mixxx package does not match the benchmark pin")
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\n"], text=True)
    runtime = {
        "mixxxPackageVersion": package_version,
        "mixxxBinarySha256": hashlib.sha256(pathlib.Path("/usr/bin/mixxx").read_bytes()).hexdigest(),
        "packagesSha256": hashlib.sha256(packages.encode()).hexdigest(),
        "packages": packages.splitlines(),
        "architecture": subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip(),
    }
    output = job_path.parent
    env = dict(os.environ, DISPLAY=":99", QT_QPA_PLATFORM="xcb", QT_QUICK_BACKEND="software", LANG="C.UTF-8")
    with (output / "xvfb.log").open("w") as log:
        display = subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1280x800x24"], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while not pathlib.Path("/tmp/.X11-unix/X99").exists():
                if display.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("isolated Xvfb display failed to start")
                time.sleep(0.05)
            captures = []
            for repetition in range(job["config"]["repetitions"]):
                for profile in job["config"]["profiles"]:
                    for offset in range(0, len(job["cases"]), 2):
                        cases = job["cases"][offset:offset + 2]
                        directory = output / f"{profile['name']}-{repetition}-{offset}"
                        results = run_batch([case["containerPath"] for case in cases], profile["fixedTempo"], directory, env)
                        for case, result in zip(cases, results):
                            captures.append(dict(case=case["name"], profile=profile["name"], repetition=repetition, analysis=result))
                        print(f"Mixxx {profile['name']} repeat {repetition + 1}: {', '.join(case['name'] for case in cases)}", flush=True)
            verify_inputs()
            (output / "capture.json").write_text(json.dumps(dict(runtime=runtime, captures=captures), indent=2) + "\n")
        finally:
            display.terminate()
            display.wait(timeout=5)
