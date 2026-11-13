# Audio8-ASR-Infinite: exported graphs and metadata

The streaming runtime expects a directory with the six ONNX graphs that the
conversion pipeline produces:

```
tower.onnx                 # audio tower (32 layers) + frame grouping + projector
frontend_trim0.onnx        # STFT/log-mel frontend, no leading trim
frontend_trim2.onnx        # same frontend, 2-frame leading trim (later windows)
decoder_0.onnx             # 12 decoder layers
decoder_1.onnx
decoder_2.onnx
```

The int8 bundle has the same names with the `.int8.onnx` suffix (the examples
copy them to the plain names, or pass `--decoder-prefix`).

## Metadata

Every graph carries ONNX metadata (`model_type=audio8-asr-infinite`, version,
maintainer, url, license, sample rate, frame length, delay). The runtime checks
`model_type` on the tower and refuses a mismatched graph instead of producing
garbage. Add it to a freshly exported directory with:

```bash
python3 scripts/audio8-asr-infinite/stamp-metadata.py \
  ./sherpa-onnx-audio8-asr-infinite-int8 --int8 --frame-len 8 --delay-ms 480
```

Weights in `.onnx.data` files are preserved.

## Verification

The PyTorch conversion and the streaming reference driver live in the
conversion repository (`audio8_infinite/reports/ref/`):

- `onnx_export.py` — exports the six graphs,
- `onnx_streaming_driver.py` — the reference streaming loop (same clocks,
  caches and prompt as the C++ runtime),
- `gear.py` — frame-length/delay presets.

The CI side runs the compiled example against the released int8 bundle, which is
the end-to-end check for the graphs.
