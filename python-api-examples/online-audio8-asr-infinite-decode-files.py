#!/usr/bin/env python3

"""
This file demonstrates how to use Audio8-ASR-Infinite, a streaming (online)
LLM ASR model, with the sherpa-onnx Python API.

Audio8-ASR-Infinite is a token-clock model: it emits one text token for every
`frame_len * 20 ms` of audio, independently of the audio content.  Unlike the
other streaming examples, this one feeds the wave file in small chunks and
prints the partial transcript while the audio is still being fed, which is what
a real streaming application does.

Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-infinite-int8.tar.bz2
tar xvf sherpa-onnx-audio8-asr-infinite-int8.tar.bz2
rm sherpa-onnx-audio8-asr-infinite-int8.tar.bz2

./python-api-examples/online-audio8-asr-infinite-decode-files.py \
  --model-dir=./sherpa-onnx-audio8-asr-infinite-int8 \
  --tokenizer=./sherpa-onnx-audio8-asr-infinite-int8/tokenizer \
  --frame-len=8 \
  --delay-ms=480 \
  ./sherpa-onnx-audio8-asr-infinite-int8/test_wavs/0.wav

The numbers below must match the exported graphs:

  --frame-len 4 = 80 ms, 6 = 120 ms, 8 = 160 ms per token
  --delay-ms must be a positive multiple of frame-len * 20 ms

If you use the 80 ms gear, pass `--frame-len 4` and point `--model-dir` at the
80 ms export of the checkpoint.

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
to download streaming pre-trained models.
"""

import argparse
import sys
import time
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="""Path to the directory holding tower.onnx,
        decoder_{0,1,2}.onnx and frontend_trim{0,2}.onnx""",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="""Path to the tokenizer directory (vocab.json, merges.txt,
        tokenizer_config.json)""",
    )

    parser.add_argument(
        "--frame-len",
        type=int,
        default=8,
        help="Tower frames per text token: 4 (80 ms), 6 (120 ms) or 8 (160 ms)",
    )

    parser.add_argument(
        "--delay-ms",
        type=int,
        default=480,
        help="Transcription delay in milliseconds. It must be a multiple of "
        "frame-len * 20 ms",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Prompt language: zh or en",
    )

    parser.add_argument(
        "--max-total-len",
        type=int,
        default=512,
        help="Number of KV cache slots of the decoder graphs",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Safety cap on the generated tokens of one segment. "
        "0 means unlimited",
    )

    parser.add_argument(
        "--enc-window",
        type=int,
        default=750,
        help="Tower attention sliding window, in tower frames (750 = 15 s)",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Number of milliseconds of audio fed to the stream at a time",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="""True to print the window schedule, the cache geometry and every
        clock step""",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="""The input sound file(s) to decode. Each file must be of WAVE
        format with a single channel, and each sample has 16-bit,
        i.e., int16_t. The sample rate of the file can be arbitrary and does
        not need to be 16 kHz""",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).exists(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models "
        "to download it"
    )


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
         normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def print_partial(printed: str, text: str) -> str:
    """Prints the part of `text` that has not been printed yet, so that the
    partial transcript grows on a single line while the audio is being fed.

    Args:
      printed:
        The text that has already been printed.
      text:
        The current partial transcript.

    Returns:
      The new value of `printed`.
    """
    if len(text) <= len(printed):
        return printed

    if printed and not text.startswith(printed):
        # The text was rewritten from scratch (it can happen after a stream
        # reset). Start a new line.
        sys.stdout.write("\n")
        printed = ""

    sys.stdout.write(text[len(printed) :])
    sys.stdout.flush()
    return text


def main():
    args = get_args()

    assert_file_exists(args.model_dir)
    assert_file_exists(args.tokenizer)

    recognizer = sherpa_onnx.OnlineRecognizer.from_audio8_asr_infinite(
        tokenizer=args.tokenizer,
        model_dir=args.model_dir,
        frame_len=args.frame_len,
        delay_ms=args.delay_ms,
        language=args.language,
        max_total_len=args.max_total_len,
        max_new_tokens=args.max_new_tokens,
        enc_window=args.enc_window,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
    )

    print("Started!")
    start_time = time.time()

    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        print(f"\n--- {wave_filename} ({duration:.2f} s) ---")

        stream = recognizer.create_stream()

        chunk = int(sample_rate * args.chunk_ms / 1000)
        printed = ""
        offset = 0

        while offset < len(samples):
            n = min(chunk, len(samples) - offset)
            stream.accept_waveform(sample_rate, samples[offset : offset + n])
            offset += n

            # One decode step per clock. While the stream is ready there is
            # always a full window of audio, so this never busy-waits.
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                printed = print_partial(printed, recognizer.get_result(stream))

        # No tail padding is needed: the model itself pads the audio with
        # `right_pad_tokens` clocks of silence once the input is finished.
        stream.input_finished()

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            printed = print_partial(printed, recognizer.get_result(stream))

        print(f"\nFinal text: {recognizer.get_result(stream)}")

    elapsed_seconds = time.time() - start_time
    rtf = elapsed_seconds / total_duration if total_duration else 0

    print(f"\nElapsed: {elapsed_seconds:.3f} s")
    print(f"Audio duration: {total_duration:.3f} s")
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
