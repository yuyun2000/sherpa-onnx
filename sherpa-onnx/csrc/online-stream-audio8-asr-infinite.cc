// sherpa-onnx/csrc/online-stream-audio8-asr-infinite.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/online-stream-audio8-asr-infinite.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

// A 10 ms block is counted as silence when its RMS is below this.  The
// reference has no energy-based VAD (endpointing is normally driven by the
// model's semantic-VAD heads, which this export does not include), so this is
// only used for `IsEndpoint()` and never for transcription.
constexpr float kSilenceRms = 0.006f;

// Number of samples of one 10 ms block at 16 kHz.
constexpr int32_t kBlockSamples = 160;

constexpr int32_t kSampleRate = 16000;

}  // namespace

OnlineStreamAudio8AsrInfinite::OnlineStreamAudio8AsrInfinite(
    const OnlineAudio8AsrInfiniteModel *model,
    const FeatureExtractorConfig &config, int64_t bos_id, int64_t lang_id,
    int64_t prompt_pad_id)
    : OnlineStream(config, nullptr),
      model_(model),
      bos_id_(bos_id),
      lang_id_(lang_id),
      prompt_pad_id_(prompt_pad_id) {
  const auto &gear = model_->Gear();

  state_.next_start = 0;
  state_.next_end = static_cast<int64_t>(gear.prefill_tokens) *
                    gear.samples_per_token;
  state_.next_tokens = gear.prefill_tokens;

  for (int32_t i = 0; i != gear.prefill_tokens; ++i) {
    if (i == 0) {
      state_.queue.push_back(bos_id_);
    } else if (i == 1) {
      state_.queue.push_back(lang_id_);
    } else {
      state_.queue.push_back(prompt_pad_id_);
    }
  }

  state_.zero_pad_end = gear.left_pad_samples;
  state_.segment_start_virtual = gear.left_pad_samples;

  state_.enc_k = model_->CreateEmptyEncCache(0);
  state_.enc_v = model_->CreateEmptyEncCache(1);
  state_.dec_k = model_->CreateEmptyDecCache(0);
  state_.dec_v = model_->CreateEmptyDecCache(1);
}

OnlineStreamAudio8AsrInfinite::~OnlineStreamAudio8AsrInfinite() = default;

void OnlineStreamAudio8AsrInfinite::AcceptWaveform(int32_t sampling_rate,
                                                   const float *waveform,
                                                   int32_t n) const {
  if (n <= 0) {
    return;
  }

  if (sampling_rate == kSampleRate) {
    state_.pcm.insert(state_.pcm.end(), waveform, waveform + n);
    state_.num_received += n;
    return;
  }

  if (sampling_rate <= 0) {
    SHERPA_ONNX_LOGE("Invalid sampling rate %d", sampling_rate);
    SHERPA_ONNX_EXIT(-1);
  }

  if (!resampler_) {
    resampler_ = std::make_unique<LinearResample>(
        sampling_rate, kSampleRate, 0.99f * 0.5f * kSampleRate, 6);
  }
  std::vector<float> samples;
  resampler_->Resample(waveform, n, false, &samples);
  state_.pcm.insert(state_.pcm.end(), samples.begin(), samples.end());
  state_.num_received += static_cast<int64_t>(samples.size());
}

void OnlineStreamAudio8AsrInfinite::InputFinished() const {
  if (state_.input_finished) {
    return;
  }

  if (resampler_) {
    std::vector<float> samples;
    resampler_->Resample(nullptr, 0, true, &samples);
    state_.pcm.insert(state_.pcm.end(), samples.begin(), samples.end());
    state_.num_received += static_cast<int64_t>(samples.size());
  }

  state_.input_finished = true;
  UpdateSilence();
}

int64_t OnlineStreamAudio8AsrInfinite::AvailableVirtualSamples() const {
  const auto &gear = model_->Gear();
  int64_t n = gear.left_pad_samples + state_.extra_pad_samples +
              state_.num_received;
  if (state_.input_finished) {
    n += static_cast<int64_t>(gear.right_pad_tokens) * gear.samples_per_token;
  }
  return n;
}

int64_t OnlineStreamAudio8AsrInfinite::VirtualPosition() const {
  const auto &gear = model_->Gear();
  return gear.left_pad_samples + state_.extra_pad_samples + state_.num_received;
}

bool OnlineStreamAudio8AsrInfinite::IsReady() const {
  const auto &gear = model_->Gear();
  if (state_.next_tokens <= 0) {
    return false;
  }
  return AvailableVirtualSamples() >=
         state_.next_end + Audio8InfiniteGear::kLookAheadSamples;
}

void OnlineStreamAudio8AsrInfinite::FillWindow(int64_t start, int64_t end,
                                               std::vector<float> *out) const {
  const auto &gear = model_->Gear();
  const int64_t n = end - start;
  out->assign(static_cast<size_t>(std::max<int64_t>(0, n)), 0.0f);

  for (int64_t v = start; v < end; ++v) {
    if (v < state_.zero_pad_end) {
      continue;  // left pad of this segment: digital silence
    }
    const int64_t user = v - gear.left_pad_samples - state_.extra_pad_samples;
    if (user >= state_.num_received) {
      continue;  // trailing padding after InputFinished()
    }
    const int64_t index = user - state_.pcm0;
    if (index < 0 || index >= static_cast<int64_t>(state_.pcm.size())) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: sample %" PRIi64 " is no longer buffered "
          "(pcm0=%" PRIi64 ", size=%zu). Please do not reset the stream so far "
          "back.",
          static_cast<int64_t>(user), static_cast<int64_t>(state_.pcm0),
          state_.pcm.size());
      SHERPA_ONNX_EXIT(-1);
    }
    (*out)[static_cast<size_t>(v - start)] =
        state_.pcm[static_cast<size_t>(index)];
  }
}

void OnlineStreamAudio8AsrInfinite::Compact() const {
  const auto &gear = model_->Gear();
  // Keep the look-back of the next window plus one full prefill window, so
  // that ResetState() can always re-arm from buffered audio.
  const int64_t keep_from_v = std::max<int64_t>(
      0, state_.next_start - Audio8InfiniteGear::kLookBackSamples -
             static_cast<int64_t>(gear.prefill_tokens) *
                 gear.samples_per_token);
  const int64_t keep_from_user =
      keep_from_v - gear.left_pad_samples - state_.extra_pad_samples;
  if (keep_from_user <= state_.pcm0) {
    return;
  }

  int64_t drop = std::min<int64_t>(
      keep_from_user - state_.pcm0,
      static_cast<int64_t>(state_.pcm.size()));
  state_.pcm.erase(state_.pcm.begin(), state_.pcm.begin() + drop);
  state_.pcm0 += drop;
}

void OnlineStreamAudio8AsrInfinite::UpdateSilence() const {
  int64_t begin = std::max<int64_t>(state_.silence_scanned_upto, state_.pcm0);
  int64_t end = state_.num_received;
  int64_t done = begin;
  while (done + kBlockSamples <= end) {
    const int64_t index = done - state_.pcm0;
    const float *p = state_.pcm.data() + index;
    double sum = 0;
    for (int32_t i = 0; i != kBlockSamples; ++i) {
      sum += static_cast<double>(p[i]) * p[i];
    }
    const float rms = static_cast<float>(std::sqrt(sum / kBlockSamples));
    if (rms < kSilenceRms) {
      state_.num_silence_samples += kBlockSamples;
    } else {
      state_.num_silence_samples = 0;
    }
    done += kBlockSamples;
  }
  state_.silence_scanned_upto = done;
}

void OnlineStreamAudio8AsrInfinite::ResetState() const {
  const auto &gear = model_->Gear();

  state_.pos_enc = 0;
  state_.pos_dec = 0;
  state_.num_steps = 0;
  state_.num_generated = 0;

  state_.enc_k = model_->CreateEmptyEncCache(0);
  state_.enc_v = model_->CreateEmptyEncCache(1);
  state_.dec_k = model_->CreateEmptyDecCache(0);
  state_.dec_v = model_->CreateEmptyDecCache(1);

  state_.queue.clear();
  for (int32_t i = 0; i != gear.prefill_tokens; ++i) {
    if (i == 0) {
      state_.queue.push_back(bos_id_);
    } else if (i == 1) {
      state_.queue.push_back(lang_id_);
    } else {
      state_.queue.push_back(prompt_pad_id_);
    }
  }

  state_.generated.clear();
  state_.token_strings.clear();
  state_.token_timestamps.clear();
  state_.token_probs.clear();
  state_.text.clear();
  state_.pending_bytes.clear();

  state_.num_silence_samples = 0;

  // Re-arm the schedule as if a new stream started now: the new segment gets
  // its own left pad of digital silence, followed by the audio that arrives
  // next.  This is what the model was trained on (every segment starts with
  // `left_pad_tokens` of silence at position 0).
  const int64_t prefill_samples =
      static_cast<int64_t>(gear.prefill_tokens) * gear.samples_per_token;
  const int64_t now = VirtualPosition();
  state_.segment_start_virtual = now;
  state_.segment_start_user = state_.num_received;
  state_.extra_pad_samples += gear.left_pad_samples;
  state_.zero_pad_end = now + gear.left_pad_samples;
  state_.next_tokens = gear.prefill_tokens;
  state_.next_start = now;
  state_.next_end = now + prefill_samples;
}

}  // namespace sherpa_onnx
