"""Small PCM scenarios with authored beat positions, separate from real music."""

from __future__ import annotations

import array
import math
import pathlib
import statistics
import sys
import wave


def generate_case(spec: dict, directory: pathlib.Path) -> dict:
    sample_rate = 24_000
    count = spec["beatCount"]
    bpm = spec["bpm"]
    pattern = spec["pattern"]
    beats = []
    time = spec["startSeconds"]
    tempi = []
    for index in range(count):
        local = bpm
        if pattern == "drift":
            local += (spec["endBpm"] - bpm) * index / (count - 1)
        tempi.append(local)
        beats.append(time)
        time += 60 / local
    duration = time
    pcm = [0.0] * math.ceil(duration * sample_rate)

    def pulse(start: float, frequency: float, amplitude: float, width: float) -> None:
        first = round(start * sample_rate)
        for offset in range(round(width * sample_rate)):
            position = first + offset
            if 0 <= position < len(pcm):
                seconds = offset / sample_rate
                pcm[position] += amplitude * math.exp(-6 * seconds / width) * math.sin(
                    2 * math.pi * frequency * seconds
                )

    for index, beat in enumerate(beats):
        if pattern == "sparse" and 16 <= index < 24:
            continue
        amplitude = 0.65
        if pattern == "half-time" and index % 2:
            amplitude = 0.16
        if pattern == "weak-intro" and index < 8:
            amplitude = 0.06
        if pattern == "syncopated":
            amplitude = 0.30
        pulse(beat, 70, amplitude, 0.08)
        if pattern != "syncopated":
            pulse(beat, 1400, amplitude * 0.35, 0.01)

    if pattern == "syncopated":
        for index in range(math.ceil(count * 4 / 3)):
            pulse(spec["startSeconds"] + index * 60 / bpm * 0.75, 1400, 0.60, 0.01)
    if pattern == "offbeat":
        for index in range(8):
            pulse(beats[index] - 30 / bpm, 1400, 0.65, 0.01)
    if pattern == "major-chords":
        chords = [(55, 59, 62), (60, 64, 67), (62, 66, 69), (55, 59, 62)]
        for index in range(len(pcm)):
            seconds = index / sample_rate
            chord = chords[min(int(seconds / (8 * 60 / bpm)), 3)]
            for note in chord:
                frequency = 440 * 2 ** ((note - 69) / 12)
                for harmonic, amplitude in [(1, 0.13), (2, 0.04), (3, 0.02)]:
                    pcm[index] += amplitude * math.sin(2 * math.pi * harmonic * frequency * seconds)

    # Quantization is part of the authored input contract; do not clip a mixture.
    peak = max(1.0, max(abs(value) for value in pcm))
    samples = array.array("h", (round(value / peak * 32767) for value in pcm))
    if sys.byteorder != "little":
        samples.byteswap()
    path = directory / (spec["name"] + ".wav")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return {
        "name": spec["name"],
        "path": str(path),
        "evidenceKind": "synthetic-authored-ground-truth",
        "category": pattern,
        "groundTruth": {
            "bpm": statistics.median(tempi),
            "bpmStatistic": "median-authored-instantaneous-tempo",
            "beats": beats,
            "key": spec.get("key"),
            "tempoMap": [dict(timeSeconds=beat, bpm=local) for beat, local in zip(beats, tempi)],
            "claimBoundary": "Authored positions include the silent breakdown grid; ambiguous pulse families remain explicit.",
        },
    }
