#!/usr/bin/env python3
"""Add the metadata that the runtime checks to an exported model directory.

The runtime refuses a tower whose `model_type` is not `audio8-asr-infinite`;
everything else (frame length, delay, cache sizes) is informational and lets the
publish step be validated without loading the graphs.

Usage::

    python3 stamp-metadata.py ./sherpa-onnx-audio8-asr-infinite [--int8]

The script edits the ONNX files in place. Weights stored in an external
`.onnx.data` file are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_meta_data(filename: Path, meta: dict) -> None:
    import onnx

    data_file = Path(str(filename) + ".data")
    model = onnx.load(str(filename), load_external_data=data_file.exists())

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)

    if data_file.exists():
        # the weights are already in memory; remove the old file, otherwise
        # onnx refuses to write over it
        data_file.unlink()
        onnx.save(
            model,
            str(filename),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file.name,
            size_threshold=1024,
        )
    else:
        onnx.save(model, str(filename))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="directory with the exported graphs")
    parser.add_argument("--int8", action="store_true",
                        help="only stamp the *.int8.onnx graphs (default: all)")
    parser.add_argument("--frame-len", type=int, default=0,
                        help="frame_len of the bundle (4, 6 or 8)")
    parser.add_argument("--delay-ms", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.model_dir)
    suffix = ".int8.onnx" if args.int8 else ".onnx"

    base = {
        "model_type": "audio8-asr-infinite",
        "version": "1",
        "maintainer": "k2-fsa",
        "url": "https://huggingface.co/Edge0/Audio8-ASR-Infinite",
        "license": "Apache-2.0",
        "sample_rate": 16000,
    }
    if args.frame_len:
        base["frame_len"] = args.frame_len
    if args.delay_ms:
        base["delay_ms"] = args.delay_ms

    if args.int8:
        graphs = [root / f"tower{suffix}"]
        graphs += sorted(root.glob(f"decoder_*{suffix}"))
        graphs += sorted(root.glob(f"frontend_trim*{suffix}"))
    else:
        graphs = sorted(root.glob("tower*.onnx"))
        graphs += sorted(root.glob("decoder_*.onnx"))
        graphs += sorted(root.glob("frontend_trim*.onnx"))

    for graph in graphs:
        if not graph.exists():
            print(f"skip {graph} (missing)")
            continue
        meta = dict(base)
        meta["graph"] = graph.name
        add_meta_data(graph, meta)
        print(f"stamped {graph}")


if __name__ == "__main__":
    main()
