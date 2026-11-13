// sherpa-onnx/csrc/online-stream-audio8-asr-infinite.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_ONLINE_STREAM_AUDIO8_ASR_INFINITE_H_
#define SHERPA_ONNX_CSRC_ONLINE_STREAM_AUDIO8_ASR_INFINITE_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

// Per-stream state of Audio8-ASR-Infinite.
//
// The model has no "chunk state" other than the KV caches: the conv frontend is
// re-run on a look-back / look-ahead PCM window for every clock, so the sample
// ring buffer is the only audio state.  All positions are absolute; a
// cache roll
// re-bases them, which is exact because the keys are rotated when they are
// written.
struct Audio8InfiniteStreamState {
  // ---- window schedule, in "virtual" samples -----------------------------
  // Virtual sample `v` is user sample `v - left_pad_samples` for
  // v >= left_pad_samples, and a zero of the left pad otherwise.
  int64_t next_start = 0;
  int64_t next_end = 0;
  int32_t next_tokens = 0;

  // ---- PCM ring ----------------------------------------------------------
  std::vector<float> pcm;
  int64_t pcm0 = 0;           // user sample index of pcm[0]
  int64_t num_received = 0;   // total user samples accepted
  bool input_finished = false;

  // A segment always starts with `left_pad_samples` of digital silence, like
  // the reference stream does at t = 0.  Every ResetState() inserts another
  // such pad, so virtual position `v` maps to user sample
  // `v - left_pad_samples - extra_pad_samples`.
  int64_t extra_pad_samples = 0;
  int64_t zero_pad_end = 0;  // virtual positions below this are zeros

  // ---- clocks ------------------------------------------------------------
  int64_t pos_enc = 0;  // absolute tower frames consumed
  int64_t pos_dec = 0;  // absolute text positions consumed
  int64_t num_steps = 0;      // number of windows decoded
  int64_t num_generated = 0;  // number of tokens sampled

  // ---- KV caches ---------------------------------------------------------
  std::vector<Ort::Value> enc_k;
  std::vector<Ort::Value> enc_v;
  std::vector<Ort::Value> dec_k;
  std::vector<Ort::Value> dec_v;

  // ---- text --------------------------------------------------------------
  std::deque<int64_t> queue;  // ids that are fed to the next decoder step
  std::vector<int64_t> generated;
  std::vector<std::string> token_strings;
  std::vector<float> token_timestamps;
  std::vector<float> token_probs;
  std::string text;
  std::string pending_bytes;

  // ---- endpointing -------------------------------------------------------
  int64_t num_silence_samples = 0;
  int64_t silence_scanned_upto = 0;  // user sample index
  int64_t segment_start_virtual = 0;
  int64_t segment_start_user = 0;  // user sample index of the segment start
};

// OnlineStream for Audio8-ASR-Infinite.
//
// Note: the model consumes raw PCM (the mel frontend is part of the exported
// ONNX bundle), so this stream keeps the samples itself and does not use the
// kaldi-native-fbank path of OnlineStream.  `AcceptWaveform` and
// `InputFinished` are virtual in OnlineStream for exactly this reason.
class OnlineStreamAudio8AsrInfinite : public OnlineStream {
 public:
  // `bos_id`, `lang_id` and `prompt_pad_id` build the initial prompt:
  // [bos, lang] + prompt_pad * (prefill_tokens - 2).  `prompt_pad_id` is the
  // streaming pad token ([STREAMING_PAD]) of the checkpoint, not the tokenizer
  // pad token (<|endoftext|>).
  OnlineStreamAudio8AsrInfinite(const OnlineAudio8AsrInfiniteModel *model,
                                const FeatureExtractorConfig &config,
                                int64_t bos_id, int64_t lang_id,
                                int64_t prompt_pad_id);

  ~OnlineStreamAudio8AsrInfinite() override;

  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const override;

  void InputFinished() const override;

  // True if the next clock window has enough audio.
  bool IsReady() const;

  // Virtual sample index of the next window that will be decoded.
  int64_t VirtualPosition() const;

  Audio8InfiniteStreamState &State() const { return state_; }

  // Number of virtual samples available for the schedule (input + the trailing
  // padding once the input is finished).
  int64_t AvailableVirtualSamples() const;

  // Drops the oldest buffered samples that can no longer be needed.
  void Compact() const;

  // Fills `out` with the PCM of the next window [start, end) in virtual
  // coordinates; zeroes the left pad.
  void FillWindow(int64_t start, int64_t end, std::vector<float> *out) const;

  // Clears the model state (caches, clocks, text) and re-arms the schedule so
  // that the new segment starts with a fresh left pad of digital silence,
  // exactly like the beginning of a stream.
  void ResetState() const;

  // Refreshes the trailing-silence counter from the buffered PCM. It is only
  // used for endpointing; the transcript never depends on it.
  void UpdateSilence() const;

 private:
  const OnlineAudio8AsrInfiniteModel *model_;
  int64_t bos_id_;
  int64_t lang_id_;
  int64_t prompt_pad_id_;

  mutable Audio8InfiniteStreamState state_;
  mutable std::unique_ptr<LinearResample> resampler_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_STREAM_AUDIO8_ASR_INFINITE_H_
