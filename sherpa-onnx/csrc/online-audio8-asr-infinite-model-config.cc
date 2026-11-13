// sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.h"

#include <initializer_list>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

// `streaming_n_left_pad_tokens_by_frame_len` of the checkpoint's config.json.
// It is part of the prompt length, so it can never be guessed.
int32_t LeftPadTokens(int32_t frame_len) {
  switch (frame_len) {
    case 4:
      return 18;
    case 6:
      return 12;
    case 8:
      return 9;
    default:
      return -1;
  }
}

// Number of text tokens of padding after the audio: num_delay_tokens + 1
// (the last clock) + `right_pad_text_tokens` (10) of the reference schedule.
constexpr int32_t kRightPadTextTokens = 10;

std::string Join(const std::string &dir, const std::string &name) {
  if (dir.empty()) {
    return name;
  }
  if (dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

}  // namespace

std::string OnlineAudio8AsrInfiniteModelConfig::ResolveEncoder() const {
  if (!encoder.empty()) {
    return encoder;
  }
  return Join(model_dir, "tower.onnx");
}

std::string OnlineAudio8AsrInfiniteModelConfig::ResolveDecoder(
    int32_t index) const {
  const std::string *p = nullptr;
  switch (index) {
    case 0:
      p = &decoder0;
      break;
    case 1:
      p = &decoder1;
      break;
    case 2:
      p = &decoder2;
      break;
    default:
      return {};
  }
  if (!p->empty()) {
    return *p;
  }
  return Join(model_dir, "decoder_" + std::to_string(index) + ".onnx");
}

std::string OnlineAudio8AsrInfiniteModelConfig::ResolveFrontend(
    int32_t trim) const {
  if (trim == 0 && !frontend_trim0.empty()) {
    return frontend_trim0;
  }
  if (trim == 2 && !frontend_trim2.empty()) {
    return frontend_trim2;
  }
  if (trim != 0 && trim != 2) {
    return {};
  }
  return Join(model_dir, "frontend_trim" + std::to_string(trim) + ".onnx");
}

bool OnlineAudio8AsrInfiniteModelConfig::IsSet() const {
  return !model_dir.empty() || !encoder.empty() || !decoder0.empty() ||
         !decoder1.empty() || !decoder2.empty();
}

void OnlineAudio8AsrInfiniteModelConfig::Register(ParseOptions *po) {
  po->Register("audio8-asr-infinite-model-dir", &model_dir,
               "Path to the directory holding tower.onnx, decoder_{0,1,2}.onnx "
               "and frontend_trim{0,2}.onnx for Audio8-ASR-Infinite. It fills "
               "in any of the per-graph paths that are left empty.");

  po->Register("audio8-asr-infinite-encoder", &encoder,
               "Path to tower.onnx for Audio8-ASR-Infinite (audio tower + "
               "frame grouping + projector)");

  po->Register("audio8-asr-infinite-decoder0", &decoder0,
               "Path to decoder_0.onnx (decoder layers 0-11) for "
               "Audio8-ASR-Infinite");

  po->Register("audio8-asr-infinite-decoder1", &decoder1,
               "Path to decoder_1.onnx (decoder layers 12-23) for "
               "Audio8-ASR-Infinite");

  po->Register("audio8-asr-infinite-decoder2", &decoder2,
               "Path to decoder_2.onnx (decoder layers 24-35 + lm_head) for "
               "Audio8-ASR-Infinite");

  po->Register("audio8-asr-infinite-frontend-trim0", &frontend_trim0,
               "Path to frontend_trim0.onnx (PCM window -> 20 ms tower frames, "
               "no trim) for Audio8-ASR-Infinite");

  po->Register("audio8-asr-infinite-frontend-trim2", &frontend_trim2,
               "Path to frontend_trim2.onnx (PCM window -> 20 ms tower frames, "
               "trim 2) for Audio8-ASR-Infinite");

  po->Register("audio8-asr-infinite-tokenizer", &tokenizer,
               "Path to the tokenizer directory of Audio8-ASR-Infinite "
               "(vocab.json, merges.txt, tokenizer_config.json)");

  po->Register("audio8-asr-infinite-frame-len", &frame_len,
               "Number of tower frames per text token for Audio8-ASR-Infinite. "
               "Valid values are 4 (80 ms), 6 (120 ms) and 8 (160 ms). It must "
               "match the exported graphs.");

  po->Register("audio8-asr-infinite-delay-ms", &delay_ms,
               "Transcription delay in milliseconds for Audio8-ASR-Infinite. "
               "It must be a positive multiple of frame_len * 20 ms.");

  po->Register("audio8-asr-infinite-max-total-len", &max_total_len,
               "Number of KV cache slots of the Audio8-ASR-Infinite decoder "
               "graphs");

  po->Register("audio8-asr-infinite-language", &language,
               "Prompt language of Audio8-ASR-Infinite: zh or en");

  po->Register("audio8-asr-infinite-enc-window", &enc_window,
               "Tower attention sliding window of Audio8-ASR-Infinite, in "
               "tower frames (750 = 15 s). It must match audio_config."
               "sliding_window of the checkpoint.");

  po->Register("audio8-asr-infinite-max-new-tokens", &max_new_tokens,
               "Safety cap on the number of tokens generated for one segment "
               "of Audio8-ASR-Infinite. 0 means unlimited.");

  po->Register("audio8-asr-infinite-debug", &debug,
               "true to log the window schedule and cache geometry of "
               "Audio8-ASR-Infinite");
}

bool OnlineAudio8AsrInfiniteModelConfig::Validate() const {
  if (model_dir.empty() && encoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please provide --audio8-asr-infinite-model-dir or "
        "--audio8-asr-infinite-encoder");
    return false;
  }

  if (!FileExists(ResolveEncoder())) {
    SHERPA_ONNX_LOGE("Audio8-ASR-Infinite tower '%s' does not exist",
                     ResolveEncoder().c_str());
    return false;
  }

  for (int32_t i = 0; i != 3; ++i) {
    if (!FileExists(ResolveDecoder(i))) {
      SHERPA_ONNX_LOGE("Audio8-ASR-Infinite decoder chunk %d '%s' does not "
                       "exist",
                       i, ResolveDecoder(i).c_str());
      return false;
    }
  }

  for (int32_t trim : {0, 2}) {
    if (!FileExists(ResolveFrontend(trim))) {
      SHERPA_ONNX_LOGE(
          "Audio8-ASR-Infinite frontend '%s' does not exist. Both "
          "frontend_trim0.onnx and frontend_trim2.onnx are required: the first "
          "window of a stream uses trim 0, every later window uses trim 2.",
          ResolveFrontend(trim).c_str());
      return false;
    }
  }

  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --audio8-asr-infinite-tokenizer");
    return false;
  }

  for (const char *name :
       {"vocab.json", "merges.txt", "tokenizer_config.json"}) {
    std::string p = tokenizer + "/" + name;
    if (!FileExists(p)) {
      SHERPA_ONNX_LOGE(
          "'%s' does not exist. Please check --audio8-asr-infinite-tokenizer. "
          "vocab.json and merges.txt can be derived from the checkpoint's "
          "tokenizer.json.",
          p.c_str());
      return false;
    }
  }

  Audio8InfiniteGear gear;
  std::string error;
  if (!Audio8InfiniteGear::FromConfig(frame_len, delay_ms, &gear, &error)) {
    SHERPA_ONNX_LOGE("Invalid Audio8-ASR-Infinite gear: %s", error.c_str());
    return false;
  }

  if (max_total_len <= 0) {
    SHERPA_ONNX_LOGE(
        "--audio8-asr-infinite-max-total-len should be > 0. Given: %d",
        max_total_len);
    return false;
  }

  if (enc_window <= 0) {
    SHERPA_ONNX_LOGE(
        "--audio8-asr-infinite-enc-window should be > 0. Given: %d",
        enc_window);
    return false;
  }

  if (language != "zh" && language != "en") {
    SHERPA_ONNX_LOGE(
        "--audio8-asr-infinite-language should be zh or en. Given: '%s'",
        language.c_str());
    return false;
  }

  if (max_new_tokens < 0) {
    SHERPA_ONNX_LOGE(
        "--audio8-asr-infinite-max-new-tokens should be >= 0. Given: %d",
        max_new_tokens);
    return false;
  }

  return true;
}

std::string OnlineAudio8AsrInfiniteModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineAudio8AsrInfiniteModelConfig(";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "encoder=\"" << ResolveEncoder() << "\", ";
  os << "decoder0=\"" << ResolveDecoder(0) << "\", ";
  os << "decoder1=\"" << ResolveDecoder(1) << "\", ";
  os << "decoder2=\"" << ResolveDecoder(2) << "\", ";
  os << "frontend_trim0=\"" << ResolveFrontend(0) << "\", ";
  os << "frontend_trim2=\"" << ResolveFrontend(2) << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "frame_len=" << frame_len << ", ";
  os << "delay_ms=" << delay_ms << ", ";
  os << "max_total_len=" << max_total_len << ", ";
  os << "language=\"" << language << "\", ";
  os << "enc_window=" << enc_window << ", ";
  os << "max_new_tokens=" << max_new_tokens << ")";

  return os.str();
}

bool Audio8InfiniteGear::FromConfig(int32_t frame_len, int32_t delay_ms,
                                    Audio8InfiniteGear *gear,
                                    std::string *error) {
  if (gear == nullptr) {
    return false;
  }

  if (frame_len != 4 && frame_len != 6 && frame_len != 8) {
    if (error) {
      *error = "--audio8-asr-infinite-frame-len must be 4, 6 or 8, given " +
               std::to_string(frame_len);
    }
    return false;
  }

  const int32_t left_pad = LeftPadTokens(frame_len);
  const int32_t clock_ms = 20 * frame_len;
  if (delay_ms <= 0 || delay_ms % clock_ms != 0) {
    if (error) {
      *error = "--audio8-asr-infinite-delay-ms must be a positive multiple of " +
               std::to_string(clock_ms) + " ms (frame_len " +
               std::to_string(frame_len) + "), given " +
               std::to_string(delay_ms);
    }
    return false;
  }

  const int32_t delay_tokens = delay_ms / clock_ms;

  gear->frame_len = frame_len;
  gear->clock_ms = clock_ms;
  gear->delay_tokens = delay_tokens;
  gear->left_pad_tokens = left_pad;
  gear->prefill_tokens = left_pad + delay_tokens + 1;
  gear->right_pad_tokens = delay_tokens + 1 + kRightPadTextTokens;
  gear->samples_per_token = 16000 * clock_ms / 1000;
  gear->left_pad_samples = left_pad * gear->samples_per_token;
  return true;
}

int32_t Audio8InfiniteGear::TowerFramesBeforeTrim(int32_t n_pcm) {
  int32_t t_mel = MelFrames(n_pcm);
  if (t_mel < 3) {
    return 0;
  }
  // conv1 keeps the length, conv2 has kernel 3 / stride 2.
  return (t_mel + 1 - 3) / 2 + 1;
}

int32_t Audio8InfiniteGear::TrimForWindow(int32_t n_pcm,
                                          int32_t num_tokens) const {
  return TowerFramesBeforeTrim(n_pcm) - num_tokens * frame_len;
}

}  // namespace sherpa_onnx
