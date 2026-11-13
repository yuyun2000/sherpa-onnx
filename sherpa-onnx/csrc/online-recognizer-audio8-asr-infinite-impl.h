// sherpa-onnx/csrc/online-recognizer-audio8-asr-infinite-impl.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_AUDIO8_ASR_INFINITE_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_AUDIO8_ASR_INFINITE_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream-audio8-asr-infinite.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

namespace sherpa_onnx {

// Streaming Audio8-ASR-Infinite: a token-clock LLM ASR.
//
// Everything is driven by the clock, not by the audio content: every
// `frame_len * 20 ms` of audio the tower consumes one new group of tower frames
// and the decoder emits exactly one token.  Tokens are only ever appended, so
// the partial text of a stream grows monotonically and `GetResult()` is cheap.
class OnlineRecognizerAudio8AsrInfiniteImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerAudio8AsrInfiniteImpl(
      const OnlineRecognizerConfig &config);

  template <typename Manager>
  OnlineRecognizerAudio8AsrInfiniteImpl(Manager *mgr,
                                        const OnlineRecognizerConfig &config);

  OnlineRecognizerAudio8AsrInfiniteImpl(
      const OnlineRecognizerAudio8AsrInfiniteImpl &) = delete;
  OnlineRecognizerAudio8AsrInfiniteImpl &operator=(
      const OnlineRecognizerAudio8AsrInfiniteImpl &) = delete;

  std::unique_ptr<OnlineStream> CreateStream() const override;

  bool IsReady(OnlineStream *s) const override;

  void DecodeStreams(OnlineStream **ss, int32_t n) const override;

  OnlineRecognizerResult GetResult(OnlineStream *s) const override;

  bool IsEndpoint(OnlineStream *s) const override;

  void Reset(OnlineStream *s) const override;

 private:
  static OnlineStreamAudio8AsrInfinite *Cast(OnlineStream *s);

  void InitFeatConfig();

  void InitPromptIds();

  void DecodeStream(OnlineStream *s) const;

  // Appends one generated token to the text of the stream.
  void EmitToken(Audio8InfiniteStreamState *state, int64_t token_id,
                 float log_prob) const;

  bool IsSkippable(int64_t token_id) const;

  OnlineRecognizerConfig config_;
  OnlineAudio8AsrInfiniteModel model_;
  std::unique_ptr<QwenAsrTokenizer> tokenizer_;
  Endpoint endpoint_;
  Audio8InfiniteGear gear_;

  int64_t bos_id_ = -1;   // <|im_start|>
  int64_t eos_id_ = -1;   // <|im_end|>
  int64_t pad_id_ = -1;   // <|endoftext|>
  int64_t language_id_ = -1;       // [LANGUAGE_ZH] or [LANGUAGE_EN]
  int64_t streaming_pad_id_ = -1;  // [STREAMING_PAD]
  int64_t streaming_word_id_ = -1;  // [STREAMING_WORD]
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_AUDIO8_ASR_INFINITE_IMPL_H_
