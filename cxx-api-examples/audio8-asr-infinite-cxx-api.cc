// cxx-api-examples/audio8-asr-infinite-cxx-api.cc
// Copyright (c)  2026  yuyun2000

//
// This file demonstrates how to use Audio8-ASR-Infinite with sherpa-onnx's C++
// API.
//
// Audio8-ASR-Infinite is a *streaming* (online) model, so this example feeds a
// wave file in 100 ms chunks, decodes after every chunk, and prints the
// partial transcript while the audio is still being fed.  The final text is
// printed once the input is finished.
//
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-infinite-0.1b-int8.tar.bz2
tar xvf sherpa-onnx-audio8-asr-infinite-0.1b-int8.tar.bz2
rm sherpa-onnx-audio8-asr-infinite-0.1b-int8.tar.bz2
*/
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

// Prints the part of `text` that has not been printed yet, so that the partial
// transcript grows on a single line while the audio is being fed.
static void PrintPartial(std::string *printed, const std::string &text) {
  if (text.size() <= printed->size()) {
    return;
  }

  size_t n_old = printed->size();
  if (n_old > 0 && text.compare(0, n_old, *printed) != 0) {
    // The text was rewritten from scratch (it can happen after a stream
    // reset).  Start a new line.
    std::cout << "\n";
    n_old = 0;
  }

  std::cout << text.substr(n_old) << std::flush;
  *printed = text;
}

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  // Number of samples of one chunk: 1600 = 100 ms at 16 kHz.
  constexpr int32_t kChunk = 1600;

  OnlineRecognizerConfig config;

  // clang-format off
  const std::string dir = "./sherpa-onnx-audio8-asr-infinite-0.1b-int8";

  // Leaving the per-graph paths empty resolves them against `dir`.
  config.model_config.audio8_asr_infinite.model_dir = dir;
  config.model_config.audio8_asr_infinite.tokenizer = dir + "/tokenizer";

  // 8 -> 160 ms per token.  The other gears of the checkpoint are 4 (80 ms)
  // and 6 (120 ms); the exported graphs must match.
  config.model_config.audio8_asr_infinite.frame_len = 8;

  // 480 ms = 3 tokens of delay at 160 ms per token.  It must be a multiple of
  // frame_len * 20 ms.
  config.model_config.audio8_asr_infinite.delay_ms = 480;

  config.model_config.audio8_asr_infinite.language = "zh";
  config.model_config.audio8_asr_infinite.max_total_len = 512;
  config.model_config.audio8_asr_infinite.enc_window = 750;
  config.model_config.audio8_asr_infinite.max_new_tokens = 0;

  // There is no tokens.txt: this model family loads its tokenizer from the
  // `tokenizer` directory.
  config.model_config.num_threads = 2;
  config.model_config.provider = "cpu";
  config.model_config.debug = false;

  // Audio8-ASR-Infinite supports only greedy_search.
  config.decoding_method = "greedy_search";

  // The acoustic endpointing fallback cuts a sentence at every long pause, so
  // it is disabled here.
  config.enable_endpoint = false;

  const std::string wave_filename = dir + "/test_wavs/0.wav";
  // clang-format on

  std::cout << "Loading model from " << dir << "\n";
  std::cout << "This model is large, please wait\n";

  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  const float duration =
      wave.samples.size() / static_cast<float>(wave.sample_rate);

  std::cout << "Started! Duration: " << duration << " s\n";

  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();

  std::string printed;
  size_t offset = 0;

  while (offset < wave.samples.size()) {
    size_t n = kChunk;
    if (offset + n > wave.samples.size()) {
      n = wave.samples.size() - offset;
    }

    stream.AcceptWaveform(wave.sample_rate, wave.samples.data() + offset,
                          static_cast<int32_t>(n));
    offset += n;

    // One decode step per clock.  While the stream is ready there is always a
    // full window of audio, so this never busy-waits.
    while (recognizer.IsReady(&stream)) {
      recognizer.Decode(&stream);

      OnlineRecognizerResult result = recognizer.GetResult(&stream);
      PrintPartial(&printed, result.text);
    }
  }

  // No tail padding is needed: the model itself pads the audio with
  // `right_pad_tokens` clocks of silence once the input is finished.
  stream.InputFinished();

  while (recognizer.IsReady(&stream)) {
    recognizer.Decode(&stream);

    OnlineRecognizerResult result = recognizer.GetResult(&stream);
    PrintPartial(&printed, result.text);
  }

  OnlineRecognizerResult result = recognizer.GetResult(&stream);

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  const float rtf = elapsed_seconds / duration;

  std::cout << "\n";
  std::cout << "Final text: " << result.text << "\n";
  std::cout << "Duration: " << duration << " s\n";
  std::cout << "Elapsed: " << elapsed_seconds << " s\n";
  std::cout << "RTF: " << rtf << "\n";

  return 0;
}
