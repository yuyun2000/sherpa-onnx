// sherpa-onnx/csrc/online-recognizer-audio8-asr-infinite-impl.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/online-recognizer-audio8-asr-infinite-impl.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model.h"
#include "sherpa-onnx/csrc/online-stream-audio8-asr-infinite.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

namespace sherpa_onnx {

namespace {

constexpr int32_t kSampleRate = 16000;

// Greedy argmax over one logits row, with `eos` forced to -inf like the
// reference driver does, plus its log-softmax probability.
int64_t SampleToken(const float *row, int32_t vocab, int64_t eos,
                    float *log_prob) {
  int64_t best = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i != vocab; ++i) {
    if (i == eos) {
      continue;
    }
    const float v = row[i];
    if (std::isfinite(v) && v > best_val) {
      best_val = v;
      best = i;
    }
  }

  double sum = 0;
  for (int32_t i = 0; i != vocab; ++i) {
    if (i == eos) {
      continue;
    }
    const float v = row[i];
    if (std::isfinite(v)) {
      sum += std::exp(static_cast<double>(v) - best_val);
    }
  }

  *log_prob = sum > 0 ? static_cast<float>(-std::log(sum)) : 0.0f;
  return best;
}

}  // namespace

OnlineRecognizerAudio8AsrInfiniteImpl::OnlineRecognizerAudio8AsrInfiniteImpl(
    const OnlineRecognizerConfig &config)
    : OnlineRecognizerImpl(config),
      config_(config),
      model_(config.model_config),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          config.model_config.audio8_asr_infinite.tokenizer)),
      endpoint_(config_.endpoint_config),
      gear_(model_.Gear()) {
  if (config.decoding_method != "greedy_search") {
    SHERPA_ONNX_LOGE(
        "Unsupported decoding method: %s. Audio8-ASR-Infinite is a token-clock "
        "LLM model and supports only greedy_search.",
        config.decoding_method.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  InitPromptIds();
  InitFeatConfig();
}

template <typename Manager>
OnlineRecognizerAudio8AsrInfiniteImpl::OnlineRecognizerAudio8AsrInfiniteImpl(
    Manager *mgr, const OnlineRecognizerConfig &config)
    : OnlineRecognizerImpl(mgr, config),
      config_(config),
      model_(mgr, config.model_config),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          mgr, config.model_config.audio8_asr_infinite.tokenizer)),
      endpoint_(config_.endpoint_config),
      gear_(model_.Gear()) {
  if (config.decoding_method != "greedy_search") {
    SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                     config.decoding_method.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  InitPromptIds();
  InitFeatConfig();
}

void OnlineRecognizerAudio8AsrInfiniteImpl::InitFeatConfig() {
  // The mel frontend is part of the exported ONNX bundle (the frontend graphs
  // consume raw PCM), so the kaldi-native-fbank extractor of OnlineStream is
  // never used.  Only the sampling rate matters, and it is 16 kHz.
  config_.feat_config.sampling_rate = kSampleRate;
  config_.feat_config.feature_dim = 128;
}

void OnlineRecognizerAudio8AsrInfiniteImpl::InitPromptIds() {
  const auto &c = config_.model_config.audio8_asr_infinite;

  bos_id_ = tokenizer_->GetTokenId("<|im_start|>");
  eos_id_ = tokenizer_->GetImEndTokenId();
  pad_id_ = tokenizer_->GetPadTokenId();
  streaming_pad_id_ = tokenizer_->GetTokenId("[STREAMING_PAD]");
  streaming_word_id_ = tokenizer_->GetTokenId("[STREAMING_WORD]");

  const std::string lang_token =
      c.language == "en" ? "[LANGUAGE_EN]" : "[LANGUAGE_ZH]";
  language_id_ = tokenizer_->GetTokenId(lang_token);

  const int32_t vocab_size = model_.VocabSize();
  if (bos_id_ < 0 || eos_id_ < 0 || pad_id_ < 0 || language_id_ < 0 ||
      streaming_pad_id_ < 0 || streaming_word_id_ < 0) {
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: the tokenizer is missing the prompt tokens. "
        "bos=%" PRIi64 " eos=%" PRIi64 " pad=%" PRIi64
        " language(%s)=%" PRIi64 " streaming_pad=%" PRIi64 " "
        "streaming_word=%" PRIi64
        ". vocab.json must contain the 26 added tokens of "
        "the checkpoint (ids 151643..151668).",
        static_cast<int64_t>(bos_id_), static_cast<int64_t>(eos_id_),
        static_cast<int64_t>(pad_id_), lang_token.c_str(),
        static_cast<int64_t>(language_id_),
        static_cast<int64_t>(streaming_pad_id_),
        static_cast<int64_t>(streaming_word_id_));
    SHERPA_ONNX_EXIT(-1);
  }

  if (bos_id_ >= vocab_size || eos_id_ >= vocab_size ||
      pad_id_ >= vocab_size || language_id_ >= vocab_size ||
      streaming_pad_id_ >= vocab_size || streaming_word_id_ >= vocab_size) {
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: a prompt token id is >= the vocabulary of the "
        "decoder (%d): bos=%" PRIi64 " eos=%" PRIi64
        " pad=%" PRIi64 " language=%" PRIi64 " "
        "streaming_pad=%" PRIi64 " streaming_word=%" PRIi64 "",
        vocab_size, static_cast<int64_t>(bos_id_),
        static_cast<int64_t>(eos_id_), static_cast<int64_t>(pad_id_),
        static_cast<int64_t>(language_id_),
        static_cast<int64_t>(streaming_pad_id_),
        static_cast<int64_t>(streaming_word_id_));
    SHERPA_ONNX_EXIT(-1);
  }

  if (c.debug || config_.model_config.debug) {
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: prompt bos=%" PRIi64
        " language=%" PRIi64 " pad=%" PRIi64 ", clock "
        "%d ms, delay %d tokens, prefill %d tokens, right pad %d tokens",
        static_cast<int64_t>(bos_id_), static_cast<int64_t>(language_id_),
        static_cast<int64_t>(pad_id_), gear_.clock_ms, gear_.delay_tokens,
        gear_.prefill_tokens, gear_.right_pad_tokens);
  }
}

OnlineStreamAudio8AsrInfinite
    *OnlineRecognizerAudio8AsrInfiniteImpl::Cast(OnlineStream *s) {
  return static_cast<OnlineStreamAudio8AsrInfinite *>(s);
}

std::unique_ptr<OnlineStream>
OnlineRecognizerAudio8AsrInfiniteImpl::CreateStream() const {
  return std::make_unique<OnlineStreamAudio8AsrInfinite>(
      &model_, config_.feat_config, bos_id_, language_id_, streaming_pad_id_);
}

bool OnlineRecognizerAudio8AsrInfiniteImpl::IsReady(OnlineStream *s) const {
  const auto &c = config_.model_config.audio8_asr_infinite;
  if (c.max_new_tokens > 0 &&
      Cast(s)->State().num_generated >= c.max_new_tokens) {
    // The safety cap stopped this segment: report "not ready" so that callers
    // do not spin on DecodeStream().
    return false;
  }

  return Cast(s)->IsReady();
}

void OnlineRecognizerAudio8AsrInfiniteImpl::DecodeStreams(OnlineStream **ss,
                                                          int32_t n) const {
  for (int32_t i = 0; i != n; ++i) {
    DecodeStream(ss[i]);
  }
}

bool OnlineRecognizerAudio8AsrInfiniteImpl::IsSkippable(
    int64_t token_id) const {
  // Mirrors the reference `decode_visible()`: the streaming pad/word tokens,
  // the bos and the pad token never reach the transcript.
  return token_id == pad_id_ || token_id == bos_id_ ||
         token_id == streaming_pad_id_ || token_id == streaming_word_id_;
}

void OnlineRecognizerAudio8AsrInfiniteImpl::EmitToken(
    Audio8InfiniteStreamState *state, int64_t token_id,
    float log_prob) const {
  if (IsSkippable(token_id)) {
    return;
  }

  std::string piece =
      tokenizer_->GetTokenStringStreaming(token_id, &state->pending_bytes);
  if (piece.empty()) {
    return;
  }

  state->text.append(piece);
  state->token_strings.push_back(std::move(piece));

  const int64_t index = state->num_generated - 1;
  const float t = static_cast<float>(index - gear_.delay_tokens) *
                  static_cast<float>(gear_.clock_ms) / 1000.0f;
  state->token_timestamps.push_back(std::max(0.0f, t));
  state->token_probs.push_back(log_prob);
}

void OnlineRecognizerAudio8AsrInfiniteImpl::DecodeStream(
    OnlineStream *s) const {
  auto *stream = Cast(s);
  auto &st = stream->State();

  const auto &c = config_.model_config.audio8_asr_infinite;
  const bool verbose = c.debug || config_.model_config.debug;

  if (!stream->IsReady()) {
    return;
  }

  stream->Compact();

  const int32_t num_tokens = st.next_tokens;
  const int32_t num_frames = num_tokens * gear_.frame_len;
  const int64_t win_start = std::max<int64_t>(
      st.next_start - Audio8InfiniteGear::kLookBackSamples, 0);
  const int64_t win_end = st.next_end + Audio8InfiniteGear::kLookAheadSamples;

  std::vector<float> window;
  stream->FillWindow(win_start, win_end, &window);
  const int32_t n_pcm = static_cast<int32_t>(window.size());

  // ---- audio tower (with a roll of the 1024-slot cache if needed) ---------
  const int64_t enc_shift = model_.EncRollShift(st.pos_enc, num_frames);
  if (enc_shift > 0) {
    model_.RollEncCache(enc_shift, &st.enc_k, &st.enc_v);
    st.pos_enc -= enc_shift;
    if (verbose) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: rolled the tower cache by %" PRIi64
          " frames, %" PRIi64 " "
          "remain",
          static_cast<int64_t>(enc_shift), static_cast<int64_t>(st.pos_enc));
    }
  }

  Ort::Value audio_embeds =
      model_.ForwardTower(window.data(), n_pcm, num_tokens, st.pos_enc,
                          &st.enc_k, &st.enc_v);
  st.pos_enc += num_frames;

  // ---- decoder -----------------------------------------------------------
  std::vector<int64_t> ids(static_cast<size_t>(num_tokens));
  for (int32_t i = 0; i != num_tokens; ++i) {
    if (!st.queue.empty()) {
      ids[i] = st.queue.front();
      st.queue.pop_front();
    } else {
      // The reference feeds the streaming pad token, not <|endoftext|>.
      ids[i] = streaming_pad_id_;
    }
  }

  const int64_t dec_shift = model_.DecRollShift(st.pos_dec, num_tokens);
  if (dec_shift > 0) {
    model_.RollDecCache(dec_shift, &st.dec_k, &st.dec_v);
    st.pos_dec -= dec_shift;
    if (verbose) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: rolled the decoder cache by %" PRIi64
          " slots, %" PRIi64 " "
          "remain",
          static_cast<int64_t>(dec_shift), static_cast<int64_t>(st.pos_dec));
    }
  }

  Ort::Value logits =
      model_.ForwardDecoder(ids.data(), num_tokens, audio_embeds, st.pos_dec,
                            &st.dec_k, &st.dec_v);
  st.pos_dec += num_tokens;

  // ---- greedily take one token from the last position --------------------
  // The last decoder chunk returns the logits of the last position only, i.e.
  // [1, 1, vocab], even when the step fed several positions (the prefill).
  const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape.back() <= 0) {
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: unexpected logits shape; expected [1, 1, vocab]");
    SHERPA_ONNX_EXIT(-1);
  }
  const int32_t vocab = static_cast<int32_t>(shape.back());
  const int64_t num_rows = shape[1];
  if (num_rows <= 0 || num_rows > num_tokens) {
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: the decoder returned %" PRIi64
        " logits rows for a step "
        "of %d tokens",
        static_cast<int64_t>(num_rows), num_tokens);
    SHERPA_ONNX_EXIT(-1);
  }
  const float *data = logits.GetTensorData<float>();
  const float *row =
      data + static_cast<size_t>(num_rows - 1) * static_cast<size_t>(vocab);

  float log_prob = 0;
  const int64_t token = SampleToken(row, vocab, eos_id_, &log_prob);

  st.queue.push_back(token);
  st.generated.push_back(token);
  st.num_generated += 1;
  st.num_steps += 1;

  EmitToken(&st, token, log_prob);

  // ---- advance the clock -------------------------------------------------
  st.next_start = st.next_end;
  st.next_end += gear_.samples_per_token;
  st.next_tokens = 1;

  stream->UpdateSilence();

  if (verbose && (st.num_steps <= 3 || st.num_steps % 10 == 0)) {
    double energy = 0;
    for (float x : window) {
      energy += static_cast<double>(x) * x;
    }
    energy = n_pcm > 0 ? std::sqrt(energy / n_pcm) : 0.0;
    SHERPA_ONNX_LOGE(
        "audio8-asr-infinite: step %" PRIi64 " window [%" PRIi64
        ", %" PRIi64 ") n_pcm=%d "
        "tokens=%d pos_enc=%" PRIi64 " pos_dec=%" PRIi64
        " token=%" PRIi64 " rms=%.4f text='%s'",
        static_cast<int64_t>(st.num_steps),
        static_cast<int64_t>(win_start), static_cast<int64_t>(win_end),
        n_pcm, num_tokens, static_cast<int64_t>(st.pos_enc),
        static_cast<int64_t>(st.pos_dec), static_cast<int64_t>(token),
        energy, st.text.c_str());
  }
}

OnlineRecognizerResult OnlineRecognizerAudio8AsrInfiniteImpl::GetResult(
    OnlineStream *s) const {
  auto *stream = Cast(s);
  const auto &st = stream->State();

  OnlineRecognizerResult r;
  r.text = st.text;
  r.tokens = st.token_strings;
  r.timestamps = st.token_timestamps;
  r.ys_probs = st.token_probs;
  r.segment = s->GetCurrentSegment();
  r.start_time = static_cast<float>(st.segment_start_user) /
                 static_cast<float>(kSampleRate);

  r.text = ApplyInverseTextNormalization(std::move(r.text));
  r.text = ApplyHomophoneReplacer(std::move(r.text));
  return r;
}

bool OnlineRecognizerAudio8AsrInfiniteImpl::IsEndpoint(OnlineStream *s) const {
  if (!config_.enable_endpoint) {
    return false;
  }

  auto *stream = Cast(s);
  const auto &st = stream->State();

  if (st.num_generated == 0) {
    return false;
  }

  // The model has no blank token: it keeps emitting tokens (often punctuation
  // or spaces) while the audio is silent, so `num_trailing_blanks` cannot be
  // used.  This fallback counts 10 ms blocks whose RMS is below the silence
  // threshold, and reuses the standard Endpoint rules on the 10 ms grid.
  const int64_t position = stream->VirtualPosition();
  int32_t num_frames = static_cast<int32_t>(
      std::max<int64_t>(0, position - st.segment_start_virtual) / 160);
  int32_t trailing_silence =
      static_cast<int32_t>(st.num_silence_samples / 160);
  trailing_silence = std::min(trailing_silence, num_frames);

  return endpoint_.IsEndpoint(num_frames, trailing_silence, 0.01f);
}

void OnlineRecognizerAudio8AsrInfiniteImpl::Reset(OnlineStream *s) const {
  auto *stream = Cast(s);
  if (!stream->State().text.empty()) {
    s->GetCurrentSegment() += 1;
  }

  stream->ResetState();
  s->Reset();
}

#if __ANDROID_API__ >= 9
template OnlineRecognizerAudio8AsrInfiniteImpl::
    OnlineRecognizerAudio8AsrInfiniteImpl(AAssetManager *mgr,
                                          const OnlineRecognizerConfig &config);
#endif

#if __OHOS__
template OnlineRecognizerAudio8AsrInfiniteImpl::
    OnlineRecognizerAudio8AsrInfiniteImpl(
        NativeResourceManager *mgr, const OnlineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
