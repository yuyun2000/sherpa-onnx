// sherpa-onnx/csrc/online-audio8-asr-infinite-model.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.h"
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

// Owns the six ONNX graphs of an Audio8-ASR-Infinite bundle (two frontend
// variants, the audio tower and the three decoder chunks) and knows the shape
// contract of their KV caches.
//
// The graphs are *full-cache* graphs: every call takes the whole cache in and
// returns the whole cache out, together with the additive attention mask and
// the absolute positions, so the host side only has to keep the cache tensors
// and build two masks per step.  Both masks only cover the slots that are
// valid, which is what makes rolling the caches (and therefore a bounded
// memory footprint) possible without touching the graphs.
class OnlineAudio8AsrInfiniteModel {
 public:
  explicit OnlineAudio8AsrInfiniteModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineAudio8AsrInfiniteModel(Manager *mgr, const OnlineModelConfig &config);

  ~OnlineAudio8AsrInfiniteModel();

  /** Run the frontend variant selected by `trim` and then the audio tower.
   *
   * @param pcm        float32, `n_pcm` samples, already in [-1, 1].
   * @param n_pcm      Number of samples of this window.
   * @param num_tokens Number of text clocks of this window.
   * @param pos_enc    Absolute position of the first tower frame of the
   *                   window; RoPE positions grow monotonically and are only
   *                   re-based when the cache is rolled.
   * @param enc_k/enc_v  One tensor per tower layer, [slots, heads, head_dim],
   *                     updated in place.
   * @return audio embeddings, float32 [1, num_tokens, decoder hidden].
   */
  Ort::Value ForwardTower(const float *pcm, int32_t n_pcm, int32_t num_tokens,
                          int64_t pos_enc, std::vector<Ort::Value> *enc_k,
                          std::vector<Ort::Value> *enc_v) const;

  /** Run one decoder step of `num_tokens` text positions.
   *
   * @param input_ids   int64, `num_tokens` ids.
   * @param audio_embeds float32 [1, num_tokens, decoder hidden].
   * @param pos_dec     Absolute position of the first text token.
   * @param dec_k/dec_v One tensor per decoder layer, [slots, kv_heads,
   *                    head_dim], updated in place.
   * @return logits, float32 [1, num_tokens, vocab_size].
   */
  Ort::Value ForwardDecoder(const int64_t *input_ids, int32_t num_tokens,
                            const Ort::Value &audio_embeds, int64_t pos_dec,
                            std::vector<Ort::Value> *dec_k,
                            std::vector<Ort::Value> *dec_v) const;

  /** Zeroed cache tensors: one [slots, heads, head_dim] tensor per layer. */
  std::vector<Ort::Value> CreateEmptyEncCache(int32_t which) const;
  std::vector<Ort::Value> CreateEmptyDecCache(int32_t which) const;

  /** Number of tower frames to drop from the cache for a window that starts at
   * absolute position `pos_enc` and adds `num_frames` frames.  Returns 0 when
   * nothing has to be rolled.
   *
   * The reference keeps the newest `EncRollKeep()` frames and re-bases the
   * absolute positions.  Re-basing is exact: the keys were rotated when they
   * were written, so attention scores only depend on position differences.
   */
  int64_t EncRollShift(int64_t pos_enc, int32_t num_frames) const;

  /** Number of text positions to drop from the decoder cache before a step of
   * `num_tokens` tokens that starts at `pos_dec`.  Returns 0 when nothing has
   * to be rolled. */
  int64_t DecRollShift(int64_t pos_dec, int32_t num_tokens) const;

  /** Drop the oldest `shift` slots of a cache, zero the freed tail. */
  void RollEncCache(int64_t shift, std::vector<Ort::Value> *enc_k,
                    std::vector<Ort::Value> *enc_v) const;
  void RollDecCache(int64_t shift, std::vector<Ort::Value> *dec_k,
                    std::vector<Ort::Value> *dec_v) const;

  // Slots the exported graphs allocate, i.e. how much they can hold without a
  // roll: 1024 tower frames (20.5 s at 50 fps) and the decoder cache size.
  int32_t EncSlots() const;
  int32_t DecSlots() const;

  int32_t NumEncLayers() const;
  int32_t NumDecLayers() const;
  int32_t EncHeads() const;
  int32_t EncHeadDim() const;
  int32_t DecKvHeads() const;
  int32_t DecHeadDim() const;
  int32_t DecHiddenSize() const;
  int32_t VocabSize() const;
  int32_t EncRollKeep() const;

  const Audio8InfiniteGear &Gear() const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_AUDIO8_ASR_INFINITE_MODEL_H_
