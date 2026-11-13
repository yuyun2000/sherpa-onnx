// sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Audio8-ASR-Infinite is an LLM-based streaming ASR model: a 32-layer Voxtral
// Realtime audio tower (1280 hidden, 32 heads, 750-tower-frame sliding window)
// followed by a 36-layer Qwen2.5-3B decoder.  It is a *token-clock* model: one
// text token is emitted every `frame_len * 20 ms` of audio, independently of
// the audio content, and `delay_ms` tells how far the token lags behind the
// audio it describes.
//
// The bundle is a set of ONNX graphs exported from one checkpoint for exactly
// one "gear" = (frame_len, delay_ms):
//
//  frontend_trim{0,2}.onnx  PCM window -> 20 ms tower frames, with the first
//                           `trim` frames dropped
//  tower.onnx               tower + frame grouping + projector, full KV cache
//                           in/out (1024 slots)
//  decoder_{0,1,2}.onnx     Qwen2.5-3B in three 12-layer chunks, full KV cache
//                           in/out (512 slots)
//
// Only one gear is supported per process, because both the projector width and
// the 36 baked per-layer `1 + adaLN(t_cond)` scale vectors depend on it.
struct OnlineAudio8AsrInfiniteModelConfig {
  // If non-empty, any of the per-graph paths below that are left empty are
  // resolved against this directory with the standard file names.
  std::string model_dir;

  // Path to tower.onnx (audio tower + frame grouping + projector).
  std::string encoder;

  // Path to the three decoder chunks, decoder_{0,1,2}.onnx.  An int8 copy of
  // the decoders works as well.
  std::string decoder0;
  std::string decoder1;
  std::string decoder2;

  // Path to the two frontend variants.  The reference window schedule only
  // ever needs trim 0 (first window of a stream) and trim 2 (every other
  // window), for every supported gear.
  std::string frontend_trim0;
  std::string frontend_trim2;

  // Path to the directory with vocab.json / merges.txt / tokenizer_config.json
  std::string tokenizer;

  // Number of tower frames per text token: 4 (80 ms), 6 (120 ms) or 8
  // (160 ms).  Must match the exported tower/decoder graphs.
  int32_t frame_len = 8;

  // Transcription delay in milliseconds.  Must be a positive multiple of the
  // clock (frame_len * 20 ms).
  int32_t delay_ms = 480;

  // Number of slots of the decoder KV cache.  It is validated against the
  // exported graphs and used to roll the decoder cache on long sessions.
  int32_t max_total_len = 512;

  // Prompt language: "zh" or "en" (the model has a [LANGUAGE_ZH] /
  // [LANGUAGE_EN] prompt token).
  std::string language = "zh";

  // Tower attention sliding window, in tower frames.  Must match
  // `audio_config.sliding_window` of the checkpoint (750 = 15 s).
  int32_t enc_window = 750;

  // Safety cap on the number of tokens generated for one segment.  0 means
  // unlimited, which is what a real streaming session wants.
  int32_t max_new_tokens = 0;

  // If true, log the resolved window schedule, cache geometry and each clock
  // step.
  bool debug = false;

  OnlineAudio8AsrInfiniteModelConfig() = default;

  OnlineAudio8AsrInfiniteModelConfig(
      const std::string &model_dir, const std::string &encoder,
      const std::string &decoder0, const std::string &decoder1,
      const std::string &decoder2, const std::string &frontend_trim0,
      const std::string &frontend_trim2, const std::string &tokenizer,
      int32_t frame_len, int32_t delay_ms, int32_t max_total_len,
      const std::string &language, int32_t enc_window,
      int32_t max_new_tokens, bool debug)
      : model_dir(model_dir),
        encoder(encoder),
        decoder0(decoder0),
        decoder1(decoder1),
        decoder2(decoder2),
        frontend_trim0(frontend_trim0),
        frontend_trim2(frontend_trim2),
        tokenizer(tokenizer),
        frame_len(frame_len),
        delay_ms(delay_ms),
        max_total_len(max_total_len),
        language(language),
        enc_window(enc_window),
        max_new_tokens(max_new_tokens),
        debug(debug) {}

  // Expands `model_dir`; returns the argument unchanged if it is already set.
  std::string ResolveEncoder() const;
  std::string ResolveDecoder(int32_t index) const;
  std::string ResolveFrontend(int32_t trim) const;

  bool IsSet() const;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

// Everything that follows from (frame_len, delay_ms) plus the checkpoint's
// `streaming_n_left_pad_tokens_by_frame_len` table.  This is the geometry the
// window schedule in the reference driver is built from.
struct Audio8InfiniteGear {
  int32_t frame_len = 8;
  int32_t clock_ms = 160;
  int32_t delay_tokens = 3;
  int32_t left_pad_tokens = 9;
  int32_t prefill_tokens = 13;
  int32_t right_pad_tokens = 14;
  int32_t samples_per_token = 2560;
  int32_t left_pad_samples = 23040;

  // 52.5 ms of look-back and 2.5 ms of look-ahead, so that the conv frontend
  // of a window sees exactly the same samples as the reference does.
  static constexpr int32_t kLookBackSamples = 840;
  static constexpr int32_t kLookAheadSamples = 40;

  // Builds the gear for the given frame length / delay.  Returns false (and
  // fills `error`) if the pair is not supported by the checkpoint.
  static bool FromConfig(int32_t frame_len, int32_t delay_ms,
                         Audio8InfiniteGear *gear, std::string *error);

  // Number of STFT frames of a window with `n_pcm` samples.
  static int32_t MelFrames(int32_t n_pcm) { return n_pcm / 160; }

  // Number of 20 ms tower frames the conv frontend produces for `n_pcm`
  // samples: conv1 (k=3, s=1) keeps the length, conv2 (k=3, s=2) halves it.
  static int32_t TowerFramesBeforeTrim(int32_t n_pcm);

  // Frames the frontend graph drops from the front of its output for a window
  // that must produce `num_tokens * frame_len` frames.
  int32_t TrimForWindow(int32_t n_pcm, int32_t num_tokens) const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_CONFIG_H_
