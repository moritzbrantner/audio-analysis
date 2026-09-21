#!/usr/bin/env python3
"""Probe vetted external DJ-corpus candidates and emit strong byte identities."""

from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "target" / "dj-goldens" / "external-probe" / "report.json"
CANDIDATES = (
    {
        "name": "friendly-evil-gangsta-synth-hip-hop",
        "sourceUrl": "https://upload.wikimedia.org/wikipedia/commons/1/10/Friendly_Evil_Gangsta_Synth_Hip_Hop.ogg",
        "provenanceUrl": "https://commons.wikimedia.org/wiki/File:Friendly_Evil_Gangsta_Synth_Hip_Hop.ogg",
        "expectedSha1": "4927325282f73c64855042153cfd15bb0ccab976",
        "expectedSize": 8_987_157,
    },
    {
        "name": "pop-rock",
        "sourceUrl": "https://upload.wikimedia.org/wikipedia/commons/4/41/Pop_rock.ogg",
        "provenanceUrl": "https://commons.wikimedia.org/wiki/File:Pop_rock.ogg",
        "expectedSha1": "db4b96c98b402749712adf5b138899f7761b30da",
        "expectedSize": 8_397_939,
    },
    {
        "name": "beethoven-pathetique-adagio",
        "sourceUrl": "https://upload.wikimedia.org/wikipedia/commons/6/63/Beethoven%2C_Sonata_No._8_in_C_Minor_Pathetique%2C_Op._13_-_II._Adagio_cantabile.ogg",
        "provenanceUrl": "https://commons.wikimedia.org/wiki/File:Beethoven,_Sonata_No._8_in_C_Minor_Pathetique,_Op._13_-_II._Adagio_cantabile.ogg",
        "expectedSha1": "d3f818a65106beef18368a64c3fdbbb3ee3df2c3",
        "expectedSize": 7_450_009,
    },
)


def probe(candidate: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        str(candidate["sourceUrl"]),
        headers={"User-Agent": "audio-analysis-dj-corpus/1.0 (+https://github.com/moritzbrantner/audio-analysis)"},
    )
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            sha1.update(chunk)
            sha256.update(chunk)

    actual_sha1 = sha1.hexdigest()
    expected_sha1 = str(candidate["expectedSha1"])
    expected_size = int(candidate["expectedSize"])
    if actual_sha1 != expected_sha1:
        raise RuntimeError(
            f"{candidate['name']}: SHA-1 {actual_sha1} != published {expected_sha1}"
        )
    if size != expected_size:
        raise RuntimeError(
            f"{candidate['name']}: size {size} != published {expected_size}"
        )

    return {
        **candidate,
        "sha1": actual_sha1,
        "sha256": sha256.hexdigest(),
        "size": size,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": "audio-analysis-dj-external-probe/v1",
        "candidates": [probe(candidate) for candidate in CANDIDATES],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
