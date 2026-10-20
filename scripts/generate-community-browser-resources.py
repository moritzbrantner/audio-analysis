#!/usr/bin/env python3
"""Development-only structural export. Trained bytes come from pinned public donors.

All retained floating tensors must match deterministic preprocessing constants;
unknown tensors fail closed. Generated templates never contain trained payloads.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser(
    description="Generate weight-free community ONNX reconstruction resources (development only)."
)
parser.add_argument("--approved-bundle", type=Path, required=True)
parser.add_argument("--public-models", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--typescript-profile", type=Path)
args = parser.parse_args()
b = args.approved_bundle
root = args.public_models
out = args.output
out.mkdir(parents=True, exist_ok=True)
DONORS = {
    "segmentation-community-1.onnx": {
        "url": "https://huggingface.co/altunenes/speaker-diarization-community-1-onnx/resolve/e2e09da94ae093a56cd5a60a09b138ae3da1959c/segmentation-community-1.onnx",
        "sha256": "62d59a487d8ba877d0bd1638c53aa06a810bdb104fb776a27ec2204521711006",
    },
    "wespeaker.onnx": {
        "url": "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx",
        "sha256": "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


for name, metadata in DONORS.items():
    require(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == metadata["sha256"],
        "donor checksum mismatch",
    )
index = {"donors": DONORS, "models": {}}


def varint(data, pos):
    v = 0
    s = 0
    while True:
        c = data[pos]
        pos += 1
        v |= (c & 127) << s
        if c < 128:
            return v, pos
        s += 7
        if s > 63:
            raise ValueError("invalid protobuf")


def fields(data, start, end):
    p = start
    while p < end:
        tag, p = varint(data, p)
        wire = tag & 7
        if wire == 2:
            n, p = varint(data, p)
            yield tag >> 3, p, p + n
            p += n
        elif wire == 0:
            _, p = varint(data, p)
        elif wire == 1:
            p += 8
        elif wire == 5:
            p += 4
        else:
            raise ValueError("wire type")
    if p != end:
        raise ValueError("invalid length")


def tensors(data):
    for f, a, z in fields(data, 0, len(data)):
        if f != 7:
            continue
        for f, a, z in fields(data, a, z):
            if f != 5:
                continue
            values = {f: (a, z) for f, a, z in fields(data, a, z)}
            na, nz = values[8]
            name = data[na:nz].decode()
            yield name, values.get(9)


def key(t):
    a = numpy_helper.to_array(t)
    return str(a.dtype), tuple(a.shape), hashlib.sha256(a.tobytes()).hexdigest()


for name, donors in [
    ("segmentation.onnx", ["segmentation-community-1.onnx"]),
    ("embedding.onnx", ["wespeaker.onnx"]),
]:
    lookup = {}
    for donor in donors:
        data = (root / donor).read_bytes()
        offsets = dict(tensors(data))
        for t in onnx.load_from_string(data).graph.initializer:
            span = offsets[t.name]
            if span:
                lookup[key(t)] = {"donor": donor, "start": span[0], "end": span[1]}
    original = (b / name).read_bytes()
    approved = json.loads((out / "pyannote_diarization_manifest.json").read_text())
    require(
        approved["artifactSetSha256"]
        == "0a12189874dace9b590af9b09ef4637552006130716db57c35d72f984b36c577",
        "unapproved artifact set",
    )
    require(
        hashlib.sha256(original).hexdigest() == approved["files"][name],
        "unapproved target model",
    )
    model = onnx.load_from_string(original)
    require(
        not model.functions and not model.graph.sparse_initializer,
        "unsupported model structure",
    )
    template = bytearray(original)
    offsets = dict(tensors(original))
    recipe = []
    for t in onnx.load_from_string(original).graph.initializer:
        if key(t) not in lookup:
            continue
        start, end = offsets[t.name]
        source = lookup[key(t)]
        require(end - start == source["end"] - source["start"], "tensor span mismatch")
        template[start:end] = bytes(end - start)
        recipe.append({"targetStart": start, "targetEnd": end, **source})
    hydrated = bytearray(template)
    for part in recipe:
        data = (root / part["donor"]).read_bytes()
        hydrated[part["targetStart"] : part["targetEnd"]] = data[
            part["start"] : part["end"]
        ]
    require(bytes(hydrated) == original, "reconstructed ONNX bytes differ")
    # The only retained floating buffers are deterministic Kaldi preprocessing.
    import numpy as np
    import torch
    from torchaudio.compliance import kaldi

    allowed = {}
    if name == "embedding.onnx":
        mel, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        allowed = {
            "window": torch.hamming_window(
                400, periodic=False, alpha=0.54, beta=0.46
            ).numpy(),
            "mel": torch.nn.functional.pad(mel, (0, 1)).numpy(),
            "onnx::STFT_6642": np.ones(512, dtype=np.float32),
        }
    for tensor in onnx.load_from_string(original).graph.initializer:
        if key(tensor) not in lookup:
            require(
                tensor.name in allowed
                and np.array_equal(numpy_helper.to_array(tensor), allowed[tensor.name]),
                "unexpected retained tensor",
            )
    for node in onnx.load_from_string(original).graph.node:
        for attribute in node.attribute:
            require(
                attribute.type
                not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS),
                "nested graph requires a new weight audit",
            )
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            values = numpy_helper.to_array(attribute.t)
            if values.dtype.kind != "f":
                continue
            if values.size == 1:
                require(
                    float(values.ravel()[0])
                    in {
                        50.0,
                        8000.0,
                        2.0,
                        float(np.float32(0.97)),
                        float(np.float32(1.1920928955078125e-7)),
                        float(np.float32(1e-8)),
                    },
                    "unexpected floating scalar",
                )
            else:
                require(
                    np.all(values == 0)
                    or (
                        values.shape == (7, 3)
                        and set(map(tuple, values.tolist()))
                        == {
                            (0, 0, 0),
                            (1, 0, 0),
                            (0, 1, 0),
                            (0, 0, 1),
                            (1, 1, 0),
                            (1, 0, 1),
                            (0, 1, 1),
                        }
                    ),
                    "unexpected floating Constant tensor",
                )
    (out / (name + ".gz")).write_bytes(gzip.compress(template, mtime=0))
    (out / (name + ".json")).write_text(
        json.dumps(
            {"targetSha256": hashlib.sha256(original).hexdigest(), "parts": recipe},
            indent=2,
        )
    )
    index["models"][name] = {
        "donor": donors[0],
        "sha256": hashlib.sha256(original).hexdigest(),
        "bytes": len(original),
        "templateSha256": hashlib.sha256(
            (out / (name + ".gz")).read_bytes()
        ).hexdigest(),
        "recipeSha256": hashlib.sha256(
            (out / (name + ".json")).read_bytes()
        ).hexdigest(),
    }
    print(
        name,
        "exactReconstruction",
        True,
        "templateGzipBytes",
        len(gzip.compress(template, mtime=0)),
        "trainedSpans",
        len(recipe),
    )
(out / "reconstruction.json").write_text(json.dumps(index, indent=2) + "\n")

if args.typescript_profile:
    import base64

    manifest = json.loads((out / "pyannote_diarization_manifest.json").read_text())
    profile = {
        "manifest": manifest,
        "donors": DONORS,
        "models": index["models"],
        "files": {
            name: (out / name).read_text()
            for name in [
                "pyannote_diarization_manifest.json",
                "clustering.json",
                "MODEL_PROVENANCE.md",
                "LICENSE.md",
            ]
        },
    }
    for name, model in profile["models"].items():
        model["templateBase64"] = base64.b64encode(
            (out / (name + ".gz")).read_bytes()
        ).decode("ascii")
        model["recipe"] = json.loads((out / (name + ".json")).read_text())
    args.typescript_profile.write_text(
        "/** Generated weight-free structure. Regenerate with scripts/generate-community-browser-resources.py. */\nexport const COMMUNITY_PROFILE = "
        + json.dumps(profile, indent=2)
        + " as const;\n"
    )
