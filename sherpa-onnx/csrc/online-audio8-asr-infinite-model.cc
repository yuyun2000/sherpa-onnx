// sherpa-onnx/csrc/online-audio8-asr-infinite-model.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Additive mask value for a slot that must not be attended to.
constexpr float kMaskNeg = -1.0e9f;

std::vector<int64_t> ShapeOf(const Ort::Value &v) {
  return v.GetTensorTypeAndShapeInfo().GetShape();
}

bool StartsWith(const std::string &s, const std::string &prefix) {
  return s.size() >= prefix.size() &&
         s.compare(0, prefix.size(), prefix) == 0;
}

// Recognizes "<prefix><index>", e.g. "k.13" / "v_out.7".
bool ParseIndexedName(const std::string &name, const std::string &prefix,
                      int32_t *index) {
  if (!StartsWith(name, prefix)) {
    return false;
  }

  const std::string rest = name.substr(prefix.size());
  if (rest.empty() || rest.size() > 2) {
    return false;
  }

  int32_t v = 0;
  for (char c : rest) {
    if (c < '0' || c > '9') {
      return false;
    }
    v = v * 10 + (c - '0');
  }
  *index = v;
  return true;
}

}  // namespace

class OnlineAudio8AsrInfiniteModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        c_(config.audio8_asr_infinite),
        env_(ORT_LOGGING_LEVEL_ERROR, "audio8-asr-infinite"),
        sess_opts_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    InitGear();

    for (int32_t trim : {0, 2}) {
      frontend_sess_[trim] = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(c_.ResolveFrontend(trim)),
          sess_opts_);
    }

    tower_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c_.ResolveEncoder()), sess_opts_);

    for (int32_t i = 0; i != 3; ++i) {
      decoder_sess_[i] = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(c_.ResolveDecoder(i)), sess_opts_);
    }

    InitFrontends();
    InitTower();
    InitDecoders();

    if (c_.debug) {
      SHERPA_ONNX_LOGE("%s", c_.ToString().c_str());
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: gear frame_len=%d clock=%d ms delay=%d tokens "
          "prefill=%d left_pad=%d right_pad=%d samples_per_token=%d",
          gear_.frame_len, gear_.clock_ms, gear_.delay_tokens,
          gear_.prefill_tokens, gear_.left_pad_tokens, gear_.right_pad_tokens,
          gear_.samples_per_token);
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: tower layers=%d slots=%d heads=%d head_dim=%d",
          enc_layers_, enc_slots_, enc_heads_, enc_head_dim_);
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder layers=%d slots=%d kv_heads=%d "
          "head_dim=%d hidden=%d vocab=%d",
          dec_layers_, dec_slots_, dec_heads_, dec_head_dim_, dec_hidden_,
          vocab_size_);
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        c_(config.audio8_asr_infinite),
        env_(ORT_LOGGING_LEVEL_ERROR, "audio8-asr-infinite"),
        sess_opts_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    InitGear();

    // NOTE: the tower alone is ~4 GB, so loading from an asset manager is
    // only practical when the graphs are small (e.g. an int8 tower).  File
    // paths are the recommended way to run this model.
    for (int32_t trim : {0, 2}) {
      auto buf = ReadFile(mgr, c_.ResolveFrontend(trim));
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read '%s'",
                         c_.ResolveFrontend(trim).c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      frontend_sess_[trim] = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
    }

    {
      auto buf = ReadFile(mgr, c_.ResolveEncoder());
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read '%s'", c_.ResolveEncoder().c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      tower_sess_ =
          std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                        sess_opts_);
    }

    for (int32_t i = 0; i != 3; ++i) {
      auto buf = ReadFile(mgr, c_.ResolveDecoder(i));
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read '%s'", c_.ResolveDecoder(i).c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      decoder_sess_[i] = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
    }

    InitFrontends();
    InitTower();
    InitDecoders();
  }

  const Audio8InfiniteGear &Gear() const { return gear_; }

  Ort::Value ForwardTower(const float *pcm, int32_t n_pcm, int32_t num_tokens,
                          int64_t pos_enc, std::vector<Ort::Value> *enc_k,
                          std::vector<Ort::Value> *enc_v) const {
    const int32_t frame_len = gear_.frame_len;
    const int32_t n_frames = num_tokens * frame_len;

    const int32_t trim = gear_.TrimForWindow(n_pcm, num_tokens);
    auto it = frontend_sess_.find(trim);
    if (it == frontend_sess_.end()) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the window needs frontend trim %d, but only "
          "trim 0 and trim 2 are exported (n_pcm=%d, num_tokens=%d). Either "
          "the audio chunks or the gear do not match the exported graphs.",
          trim, n_pcm, num_tokens);
      SHERPA_ONNX_EXIT(-1);
    }

    // ---- frontend: PCM window -> 20 ms tower frames ------------------------
    std::vector<float> window(pcm, pcm + n_pcm);
    std::array<int64_t, 2> pcm_shape{1, static_cast<int64_t>(n_pcm)};
    std::vector<Ort::Value> femp;
    femp.push_back(Ort::Value::CreateTensor<float>(
        cpu_mem_info_, window.data(), window.size(), pcm_shape.data(),
        pcm_shape.size()));

    const auto &fin_ptr = frontend_input_ptrs_.at(trim);
    const auto &fout_ptr = frontend_output_ptrs_.at(trim);
    auto frames_out =
        it->second->Run(Ort::RunOptions{nullptr}, fin_ptr.data(), femp.data(),
                        femp.size(), fout_ptr.data(), fout_ptr.size());
    Ort::Value frames = std::move(frames_out[0]);

    {
      auto s = ShapeOf(frames);
      if (s.size() != 3 ||
          s[1] != static_cast<int64_t>(n_frames)) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: frontend trim %d produced %s frames, but "
            "%d were expected for %d tokens of frame_len %d",
            trim, ShapeToString(s).c_str(), n_frames, num_tokens, frame_len);
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // ---- tower -------------------------------------------------------------
    std::vector<int64_t> pos(static_cast<size_t>(n_frames));
    for (int32_t i = 0; i != n_frames; ++i) {
      pos[i] = pos_enc + i;
    }
    std::vector<int64_t> pos_shape{1, static_cast<int64_t>(n_frames)};

    std::vector<float> mask =
        BuildEncMask(pos_enc, n_frames, enc_slots_, c_.enc_window);
    std::vector<int64_t> mask_shape{1, 1, static_cast<int64_t>(n_frames),
                                    static_cast<int64_t>(enc_slots_)};

    std::vector<Ort::Value> inputs;
    inputs.reserve(tower_input_names_.size());
    for (const auto &name : tower_input_names_) {
      int32_t layer = -1;
      if (name == "frames") {
        inputs.push_back(std::move(frames));
      } else if (name == "pos") {
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu_mem_info_, pos.data(), pos.size(), pos_shape.data(),
            pos_shape.size()));
      } else if (name == "mask") {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu_mem_info_, mask.data(), mask.size(), mask_shape.data(),
            mask_shape.size()));
      } else if (ParseIndexedName(name, "k.", &layer) ||
                 ParseIndexedName(name, "v.", &layer)) {
        // Filled below from the cache; the placeholder keeps the order.
        inputs.emplace_back(nullptr);
      } else {
        SHERPA_ONNX_LOGE("audio8-asr-infinite: unexpected tower input '%s'",
                         name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // The cache tensors are moved into the input list: the graph returns the
    // whole cache again, so the previous value is dead after this call.
    for (int32_t i = 0; i != enc_layers_; ++i) {
      inputs[IndexByName(tower_input_names_, "k." + std::to_string(i))] =
          std::move((*enc_k)[i]);
    }
    for (int32_t i = 0; i != enc_layers_; ++i) {
      inputs[IndexByName(tower_input_names_, "v." + std::to_string(i))] =
          std::move((*enc_v)[i]);
    }

    auto outputs =
        tower_sess_->Run(Ort::RunOptions{nullptr}, tower_input_ptrs_.data(),
                         inputs.data(), inputs.size(),
                         tower_output_ptrs_.data(), tower_output_ptrs_.size());

    Ort::Value audio_embeds;
    for (size_t i = 0; i != tower_output_names_.size(); ++i) {
      const auto &name = tower_output_names_[i];
      int32_t layer = -1;
      if (name == tower_audio_output_name_) {
        audio_embeds = std::move(outputs[i]);
      } else if (ParseIndexedName(name, "k_out.", &layer)) {
        (*enc_k)[layer] = std::move(outputs[i]);
      } else if (ParseIndexedName(name, "v_out.", &layer)) {
        (*enc_v)[layer] = std::move(outputs[i]);
      } else {
        SHERPA_ONNX_LOGE("audio8-asr-infinite: unexpected tower output '%s'",
                         name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    return audio_embeds;
  }

  Ort::Value ForwardDecoder(const int64_t *input_ids, int32_t num_tokens,
                            const Ort::Value &audio_embeds, int64_t pos_dec,
                            std::vector<Ort::Value> *dec_k,
                            std::vector<Ort::Value> *dec_v) const {
    std::vector<int64_t> pos(static_cast<size_t>(num_tokens));
    for (int32_t i = 0; i != num_tokens; ++i) {
      pos[i] = pos_dec + i;
    }
    std::vector<int64_t> pos_shape{1, static_cast<int64_t>(num_tokens)};
    std::vector<int64_t> ids(input_ids, input_ids + num_tokens);
    std::vector<int64_t> ids_shape{1, static_cast<int64_t>(num_tokens)};

    std::vector<float> mask = BuildDecMask(pos_dec, num_tokens, dec_slots_);
    std::vector<int64_t> mask_shape{1, 1, static_cast<int64_t>(num_tokens),
                                    static_cast<int64_t>(dec_slots_)};

    Ort::Value hidden;
    Ort::Value logits;

    for (int32_t chunk = 0; chunk != 3; ++chunk) {
      const int32_t layer_begin = dec_chunk_begin_[chunk];
      const int32_t layer_end = layer_begin + dec_chunk_layers_[chunk];

      std::vector<Ort::Value> inputs;
      inputs.reserve(decoder_input_names_[chunk].size());
      for (const auto &name : decoder_input_names_[chunk]) {
        if (name == "input_ids") {
          inputs.push_back(Ort::Value::CreateTensor<int64_t>(
              cpu_mem_info_, ids.data(), ids.size(), ids_shape.data(),
              ids_shape.size()));
        } else if (name == "audio_embeds") {
          inputs.push_back(Clone(allocator_, &audio_embeds));
        } else if (StartsWith(name, "hidden")) {
          inputs.push_back(std::move(hidden));
        } else if (name == "pos") {
          inputs.push_back(Ort::Value::CreateTensor<int64_t>(
              cpu_mem_info_, pos.data(), pos.size(), pos_shape.data(),
              pos_shape.size()));
        } else if (name == "mask") {
          inputs.push_back(Ort::Value::CreateTensor<float>(
              cpu_mem_info_, mask.data(), mask.size(), mask_shape.data(),
              mask_shape.size()));
        } else {
          int32_t layer = -1;
          if (ParseIndexedName(name, "k.", &layer) ||
              ParseIndexedName(name, "v.", &layer)) {
            // Filled below from the cache; the placeholder keeps the order.
            inputs.emplace_back(nullptr);
            continue;
          }
          SHERPA_ONNX_LOGE("audio8-asr-infinite: unexpected decoder input '%s'",
                           name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }

      for (int32_t i = layer_begin; i != layer_end; ++i) {
        inputs[IndexByName(decoder_input_names_[chunk], "k." +
                                                           std::to_string(i))] =
            std::move((*dec_k)[i]);
      }
      for (int32_t i = layer_begin; i != layer_end; ++i) {
        inputs[IndexByName(decoder_input_names_[chunk], "v." +
                                                           std::to_string(i))] =
            std::move((*dec_v)[i]);
      }

      auto outputs = decoder_sess_[chunk]->Run(
          Ort::RunOptions{nullptr}, decoder_input_ptrs_[chunk].data(),
          inputs.data(), inputs.size(), decoder_output_ptrs_[chunk].data(),
          decoder_output_ptrs_[chunk].size());

      for (size_t i = 0; i != decoder_output_names_[chunk].size(); ++i) {
        const auto &name = decoder_output_names_[chunk][i];
        int32_t layer = -1;
        if (name == "logits") {
          logits = std::move(outputs[i]);
        } else if (name == "hidden") {
          hidden = std::move(outputs[i]);
        } else if (ParseIndexedName(name, "k_out.", &layer)) {
          (*dec_k)[layer] = std::move(outputs[i]);
        } else if (ParseIndexedName(name, "v_out.", &layer)) {
          (*dec_v)[layer] = std::move(outputs[i]);
        } else {
          SHERPA_ONNX_LOGE(
              "audio8-asr-infinite: unexpected decoder output '%s'",
              name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }
    }

    return logits;
  }

  std::vector<Ort::Value> CreateEmptyEncCache(int32_t which) const {
    return CreateEmptyCache(enc_layers_, enc_slots_, enc_heads_, enc_head_dim_,
                            which);
  }

  std::vector<Ort::Value> CreateEmptyDecCache(int32_t which) const {
    return CreateEmptyCache(dec_layers_, dec_slots_, dec_heads_, dec_head_dim_,
                            which);
  }

  int64_t EncRollShift(int64_t pos_enc, int32_t num_frames) const {
    if (pos_enc + num_frames <= enc_slots_) {
      return 0;
    }

    const int64_t keep = EncRollKeep();
    int64_t shift = (pos_enc - keep) / gear_.frame_len * gear_.frame_len;
    if (shift <= 0) {
      return 0;
    }

    if (shift >= pos_enc) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower cache needs a roll of %" PRIi64
          " frames "
          "but only %" PRIi64 " are cached",
          static_cast<int64_t>(shift), static_cast<int64_t>(pos_enc));
      SHERPA_ONNX_EXIT(-1);
    }
    return shift;
  }

  int64_t DecRollShift(int64_t pos_dec, int32_t num_tokens) const {
    if (pos_dec + num_tokens <= dec_slots_) {
      return 0;
    }

    int64_t keep = c_.max_total_len > 0 ? c_.max_total_len : dec_slots_;
    keep = std::min<int64_t>(keep, dec_slots_);
    // Must leave room for this step.
    keep = std::min<int64_t>(keep, dec_slots_ - num_tokens);
    if (keep <= 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the decoder window (%d) is too small for a "
          "step of %d tokens with a cache of %d slots",
          c_.max_total_len, num_tokens, dec_slots_);
      SHERPA_ONNX_EXIT(-1);
    }

    int64_t shift = pos_dec + num_tokens - keep;
    if (shift <= 0) {
      return 0;
    }
    if (shift >= pos_dec) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the decoder cache needs a roll of %" PRIi64
          " slots "
          "but only %" PRIi64 " are cached",
          static_cast<int64_t>(shift), static_cast<int64_t>(pos_dec));
      SHERPA_ONNX_EXIT(-1);
    }
    return shift;
  }

  void RollEncCache(int64_t shift, std::vector<Ort::Value> *enc_k,
                    std::vector<Ort::Value> *enc_v) const {
    RollCache(shift, enc_slots_, enc_heads_, enc_head_dim_, enc_k);
    RollCache(shift, enc_slots_, enc_heads_, enc_head_dim_, enc_v);
  }

  void RollDecCache(int64_t shift, std::vector<Ort::Value> *dec_k,
                    std::vector<Ort::Value> *dec_v) const {
    RollCache(shift, dec_slots_, dec_heads_, dec_head_dim_, dec_k);
    RollCache(shift, dec_slots_, dec_heads_, dec_head_dim_, dec_v);
  }

  int32_t EncSlots() const { return enc_slots_; }
  int32_t DecSlots() const { return dec_slots_; }
  int32_t NumEncLayers() const { return enc_layers_; }
  int32_t NumDecLayers() const { return dec_layers_; }
  int32_t EncHeads() const { return enc_heads_; }
  int32_t EncHeadDim() const { return enc_head_dim_; }
  int32_t DecKvHeads() const { return dec_heads_; }
  int32_t DecHeadDim() const { return dec_head_dim_; }
  int32_t DecHiddenSize() const { return dec_hidden_; }
  int32_t VocabSize() const { return vocab_size_; }
  // The reference keeps the newest 896 tower frames (>= the 750-frame sliding
  // window) when its 1024-slot cache overflows.
  int32_t EncRollKeep() const { return enc_keep_; }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void InitGear() {
    std::string error;
    if (!Audio8InfiniteGear::FromConfig(c_.frame_len, c_.delay_ms, &gear_,
                                       &error)) {
      SHERPA_ONNX_LOGE("audio8-asr-infinite: %s", error.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  static std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::string s = "[";
    for (size_t i = 0; i != shape.size(); ++i) {
      s += std::to_string(shape[i]);
      if (i + 1 != shape.size()) {
        s += ",";
      }
    }
    s += "]";
    return s;
  }

  static size_t IndexByName(const std::vector<std::string> &names,
                            const std::string &name) {
    for (size_t i = 0; i != names.size(); ++i) {
      if (names[i] == name) {
        return i;
      }
    }
    SHERPA_ONNX_LOGE("audio8-asr-infinite: the graphs are missing '%s'",
                     name.c_str());
    SHERPA_ONNX_EXIT(-1);
    return 0;
  }

  void InitFrontends() {
    for (int32_t trim : {0, 2}) {
      auto &sess = frontend_sess_[trim];
      auto &in_names = frontend_input_names_[trim];
      auto &in_ptrs = frontend_input_ptrs_[trim];
      auto &out_names = frontend_output_names_[trim];
      auto &out_ptrs = frontend_output_ptrs_[trim];
      GetInputNames(sess.get(), &in_names, &in_ptrs);
      GetOutputNames(sess.get(), &out_names, &out_ptrs);

      if (in_names.size() != 1 || in_names[0] != "pcm") {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: frontend trim %d must have exactly one "
            "input named 'pcm', got %zu inputs, first '%s'",
            trim, in_names.size(), in_names.empty() ? "" : in_names[0].c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      if (out_names.size() != 1) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: frontend trim %d must have exactly one "
            "output, got %zu",
            trim, out_names.size());
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  void InitTower() {
    GetInputNames(tower_sess_.get(), &tower_input_names_,
                  &tower_input_ptrs_);
    GetOutputNames(tower_sess_.get(), &tower_output_names_,
                   &tower_output_ptrs_);

    bool has_frames = false;
    bool has_pos = false;
    bool has_mask = false;
    int32_t max_k = -1;
    int32_t max_v = -1;
    for (const auto &name : tower_input_names_) {
      if (name == "frames") {
        has_frames = true;
      } else if (name == "pos") {
        has_pos = true;
      } else if (name == "mask") {
        has_mask = true;
      } else {
        int32_t layer = -1;
        if (ParseIndexedName(name, "k.", &layer)) {
          max_k = std::max(max_k, layer);
        } else if (ParseIndexedName(name, "v.", &layer)) {
          max_v = std::max(max_v, layer);
        } else {
          SHERPA_ONNX_LOGE(
              "audio8-asr-infinite: unexpected tower input '%s'. Expected "
              "'frames', 'pos', 'mask', 'k.0..k.N-1' and 'v.0..v.N-1'.",
              name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }
    }

    if (!has_frames || !has_pos || !has_mask) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower graph must have 'frames', 'pos' and "
          "'mask' inputs");
      SHERPA_ONNX_EXIT(-1);
    }

    enc_layers_ = max_k + 1;
    if (enc_layers_ <= 0 || max_v + 1 != enc_layers_) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower has %d key tensors and %d value "
          "tensors; both must be equal and > 0",
          enc_layers_, max_v + 1);
      SHERPA_ONNX_EXIT(-1);
    }

    if (tower_input_names_.size() != static_cast<size_t>(3 + 2 * enc_layers_)) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower has %zu inputs, expected 3 + 2 * %d "
          "(frames, pos, mask, k.*, v.*)",
          tower_input_names_.size(), enc_layers_);
      SHERPA_ONNX_EXIT(-1);
    }

    // Cache geometry from k.0: [slots, heads, head_dim].
    auto kv_shape =
        tower_sess_->GetInputTypeInfo(IndexByName(tower_input_names_, "k.0"))
            .GetTensorTypeAndShapeInfo()
            .GetShape();
    if (kv_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower cache must be 3-D [slots, heads, "
          "head_dim], got %s",
          ShapeToString(kv_shape).c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    enc_slots_ = static_cast<int32_t>(kv_shape[0]);
    enc_heads_ = static_cast<int32_t>(kv_shape[1]);
    enc_head_dim_ = static_cast<int32_t>(kv_shape[2]);
    if (enc_slots_ <= 0 || enc_heads_ <= 0 || enc_head_dim_ <= 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower cache shape %s has dynamic "
          "dimensions; re-export the tower with a static cache",
          ShapeToString(kv_shape).c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    if (enc_slots_ <= c_.enc_window) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower cache has %d slots but "
          "--audio8-asr-infinite-enc-window is %d; the sliding window must be "
          "smaller than the cache",
          enc_slots_, c_.enc_window);
      SHERPA_ONNX_EXIT(-1);
    }
    // The reference keeps the newest 896 frames when its 1024-slot cache
    // overflows; a roll must always leave the whole sliding window valid.
    enc_keep_ = std::min<int32_t>(enc_slots_ - gear_.frame_len, 896);
    if (enc_keep_ < c_.enc_window + gear_.frame_len) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower cache (%d slots) is too small for a "
          "sliding window of %d frames on a %d-frame clock",
          enc_slots_, c_.enc_window, gear_.frame_len);
      SHERPA_ONNX_EXIT(-1);
    }

    bool has_audio_out = false;
    max_k = -1;
    max_v = -1;
    for (const auto &name : tower_output_names_) {
      int32_t layer = -1;
      if (ParseIndexedName(name, "k_out.", &layer)) {
        max_k = std::max(max_k, layer);
        continue;
      }
      if (ParseIndexedName(name, "v_out.", &layer)) {
        max_v = std::max(max_v, layer);
        continue;
      }
      if (name.find("audio") != std::string::npos) {
        has_audio_out = true;
        tower_audio_output_name_ = name;
        continue;
      }
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: unexpected tower output '%s'. Expected "
          "'audio_embed', 'k_out.0..k_out.N-1' and 'v_out.0..v_out.N-1'.",
          name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!has_audio_out || max_k + 1 != enc_layers_ ||
        max_v + 1 != enc_layers_) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower has %d inputs layers but %d key "
          "outputs and %d value outputs (audio output: %s)",
          enc_layers_, max_k + 1, max_v + 1, has_audio_out ? "yes" : "no");
      SHERPA_ONNX_EXIT(-1);
    }

    // The audio embeddings go straight to the decoder, so the last dim must
    // match the decoder hidden size; that check is done in InitDecoders().
    auto audio_shape =
        tower_sess_
            ->GetOutputTypeInfo(
                IndexByName(tower_output_names_, tower_audio_output_name_))
            .GetTensorTypeAndShapeInfo()
            .GetShape();
    if (audio_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower audio output must be 3-D [1, "
          "tokens, hidden], got %s",
          ShapeToString(audio_shape).c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    audio_hidden_ = static_cast<int32_t>(audio_shape.back());
  }

  void InitDecoders() {
    int32_t next_layer = 0;
    for (int32_t chunk = 0; chunk != 3; ++chunk) {
      auto &sess = decoder_sess_[chunk];
      GetInputNames(sess.get(), &decoder_input_names_[chunk],
                    &decoder_input_ptrs_[chunk]);
      GetOutputNames(sess.get(), &decoder_output_names_[chunk],
                     &decoder_output_ptrs_[chunk]);

      bool has_pos = false;
      bool has_mask = false;
      int32_t min_layer = std::numeric_limits<int32_t>::max();
      int32_t max_layer = -1;
      int32_t max_v = -1;
      int32_t num_k = 0;
      int32_t num_v = 0;
      bool has_hidden_in = false;
      bool has_ids = false;
      bool has_audio = false;
      for (const auto &name : decoder_input_names_[chunk]) {
        int32_t layer = -1;
        if (name == "input_ids") {
          has_ids = true;
        } else if (name == "audio_embeds") {
          has_audio = true;
        } else if (StartsWith(name, "hidden")) {
          has_hidden_in = true;
        } else if (name == "pos") {
          has_pos = true;
        } else if (name == "mask") {
          has_mask = true;
        } else if (ParseIndexedName(name, "k.", &layer)) {
          min_layer = std::min(min_layer, layer);
          max_layer = std::max(max_layer, layer);
          ++num_k;
        } else if (ParseIndexedName(name, "v.", &layer)) {
          max_v = std::max(max_v, layer);
          ++num_v;
        } else {
          SHERPA_ONNX_LOGE(
              "audio8-asr-infinite: unexpected decoder chunk %d input '%s'",
              chunk, name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }

      if (!has_pos || !has_mask) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d must have 'pos' and 'mask' "
            "inputs",
            chunk);
        SHERPA_ONNX_EXIT(-1);
      }
      if (chunk == 0) {
        if (!has_ids || !has_audio) {
          SHERPA_ONNX_LOGE(
              "audio8-asr-infinite: decoder chunk 0 must take 'input_ids' and "
              "'audio_embeds'");
          SHERPA_ONNX_EXIT(-1);
        }
      } else if (!has_hidden_in || has_ids || has_audio) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d must take a hidden-state "
            "input and no token/audio input",
            chunk);
        SHERPA_ONNX_EXIT(-1);
      }

      if (num_k != num_v || num_k <= 0) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d has %d key and %d value "
            "tensors",
            chunk, num_k, num_v);
        SHERPA_ONNX_EXIT(-1);
      }
      if (min_layer != next_layer || max_layer - min_layer + 1 != num_k) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d covers layers %d..%d, but "
            "layer %d is expected next; the three chunks must be contiguous "
            "and ordered",
            chunk, min_layer, max_layer, next_layer);
        SHERPA_ONNX_EXIT(-1);
      }
      if (max_v + 1 != max_layer + 1) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d layer numbering is "
            "inconsistent between keys and values",
            chunk);
        SHERPA_ONNX_EXIT(-1);
      }

      dec_chunk_begin_[chunk] = min_layer;
      dec_chunk_layers_[chunk] = num_k;

      // Cache geometry from the first key of this chunk.
      auto kv_shape =
          sess->GetInputTypeInfo(
                  IndexByName(decoder_input_names_[chunk],
                              "k." + std::to_string(min_layer)))
              .GetTensorTypeAndShapeInfo()
              .GetShape();
      if (kv_shape.size() != 3) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: the decoder cache must be 3-D [slots, "
            "kv_heads, head_dim], got %s",
            ShapeToString(kv_shape).c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      const int32_t slots = static_cast<int32_t>(kv_shape[0]);
      const int32_t heads = static_cast<int32_t>(kv_shape[1]);
      const int32_t head_dim = static_cast<int32_t>(kv_shape[2]);
      if (slots <= 0 || heads <= 0 || head_dim <= 0) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d cache shape %s has dynamic "
            "dimensions",
            chunk, ShapeToString(kv_shape).c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      if (chunk == 0) {
        dec_slots_ = slots;
        dec_heads_ = heads;
        dec_head_dim_ = head_dim;
      } else if (slots != dec_slots_ || heads != dec_heads_ ||
                 head_dim != dec_head_dim_) {
        SHERPA_ONNX_LOGE(
            "audio8-asr-infinite: decoder chunk %d cache is %s but chunk 0 is "
            "[%d, %d, %d]",
            chunk, ShapeToString(kv_shape).c_str(), dec_slots_, dec_heads_,
            dec_head_dim_);
        SHERPA_ONNX_EXIT(-1);
      }

      next_layer = max_layer + 1;
    }

    dec_layers_ = next_layer;
    if (dec_layers_ <= 0) {
      SHERPA_ONNX_LOGE("audio8-asr-infinite: no decoder layers found");
      SHERPA_ONNX_EXIT(-1);
    }

    // The last chunk produces the logits.
    const auto &last_outputs = decoder_output_names_[2];
    bool has_logits = false;
    for (const auto &name : last_outputs) {
      if (name == "logits") {
        has_logits = true;
      }
    }
    if (!has_logits) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder chunk 2 must have a 'logits' output");
      SHERPA_ONNX_EXIT(-1);
    }
    auto logits_shape =
        decoder_sess_[2]
            ->GetOutputTypeInfo(IndexByName(last_outputs, "logits"))
            .GetTensorTypeAndShapeInfo()
            .GetShape();
    if (logits_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder logits must be 3-D [1, tokens, vocab], "
          "got %s",
          ShapeToString(logits_shape).c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    vocab_size_ = static_cast<int32_t>(logits_shape.back());
    if (vocab_size_ <= 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder logits last dim is dynamic");
      SHERPA_ONNX_EXIT(-1);
    }

    // hidden size of the decoder (input of chunk 1 / chunk 2)
    auto hidden_shape =
        decoder_sess_[1]
            ->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo()
            .GetShape();
    if (hidden_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder hidden states must be 3-D, got %s",
          ShapeToString(hidden_shape).c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    dec_hidden_ = static_cast<int32_t>(hidden_shape.back());
    if (dec_hidden_ <= 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: decoder hidden size is dynamic");
      SHERPA_ONNX_EXIT(-1);
    }

    if (audio_hidden_ != dec_hidden_) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: the tower outputs %d-dim audio embeddings but "
          "the decoder expects %d; the tower and the decoder come from "
          "different exports",
          audio_hidden_, dec_hidden_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (c_.max_total_len > dec_slots_) {
      SHERPA_ONNX_LOGE(
          "audio8-asr-infinite: --audio8-asr-infinite-max-total-len is %d but "
          "the decoder cache has only %d slots",
          c_.max_total_len, dec_slots_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::vector<Ort::Value> CreateEmptyCache(int32_t layers, int32_t slots,
                                           int32_t heads,
                                           int32_t head_dim,
                                           int32_t /*which*/) const {
    std::array<int64_t, 3> shape{slots, heads, head_dim};
    const size_t num_floats =
        static_cast<size_t>(slots) * heads * head_dim;
    std::vector<Ort::Value> ans;
    ans.reserve(layers);
    for (int32_t i = 0; i != layers; ++i) {
      auto t = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                               shape.size());
      // Slots the mask hides are still read by the graph, so the cache has to
      // start from zeros.
      std::fill_n(t.GetTensorMutableData<float>(), num_floats, 0.0f);
      ans.push_back(std::move(t));
    }
    return ans;
  }

  static std::vector<float> BuildEncMask(int64_t pos_first, int32_t num_frames,
                                         int32_t slots, int32_t window) {
    std::vector<float> mask(
        static_cast<size_t>(num_frames) * slots, kMaskNeg);
    for (int32_t t = 0; t != num_frames; ++t) {
      const int64_t q = pos_first + t;
      float *row = mask.data() + static_cast<size_t>(t) * slots;
      const int64_t lo = std::max<int64_t>(0, q - window + 1);
      for (int64_t j = lo; j <= q && j < slots; ++j) {
        row[j] = 0.0f;
      }
    }
    return mask;
  }

  static std::vector<float> BuildDecMask(int64_t pos_first, int32_t num_tokens,
                                         int32_t slots) {
    std::vector<float> mask(
        static_cast<size_t>(num_tokens) * slots, kMaskNeg);
    for (int32_t t = 0; t != num_tokens; ++t) {
      const int64_t q = pos_first + t;
      float *row = mask.data() + static_cast<size_t>(t) * slots;
      for (int64_t j = 0; j <= q && j < slots; ++j) {
        row[j] = 0.0f;
      }
    }
    return mask;
  }

  static void RollCache(int64_t shift, int32_t slots, int32_t heads,
                        int32_t head_dim, std::vector<Ort::Value> *cache) {
    const size_t row = static_cast<size_t>(heads) * head_dim;
    const size_t keep_rows = static_cast<size_t>(slots - shift);
    const size_t keep_bytes = keep_rows * row * sizeof(float);
    for (auto &v : *cache) {
      float *p = v.GetTensorMutableData<float>();
      std::memmove(p, p + static_cast<size_t>(shift) * row, keep_bytes);
      std::fill_n(p + keep_rows * row,
                  static_cast<size_t>(shift) * row, 0.0f);
    }
  }

 private:
  OnlineModelConfig config_;
  OnlineAudio8AsrInfiniteModelConfig c_;
  Audio8InfiniteGear gear_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo cpu_mem_info_;

  std::map<int32_t, std::unique_ptr<Ort::Session>> frontend_sess_;
  std::unique_ptr<Ort::Session> tower_sess_;
  std::array<std::unique_ptr<Ort::Session>, 3> decoder_sess_;

  std::map<int32_t, std::vector<std::string>> frontend_input_names_;
  std::map<int32_t, std::vector<const char *>> frontend_input_ptrs_;
  std::map<int32_t, std::vector<std::string>> frontend_output_names_;
  std::map<int32_t, std::vector<const char *>> frontend_output_ptrs_;

  std::vector<std::string> tower_input_names_;
  std::vector<const char *> tower_input_ptrs_;
  std::vector<std::string> tower_output_names_;
  std::vector<const char *> tower_output_ptrs_;
  std::string tower_audio_output_name_;

  std::array<std::vector<std::string>, 3> decoder_input_names_;
  std::array<std::vector<const char *>, 3> decoder_input_ptrs_;
  std::array<std::vector<std::string>, 3> decoder_output_names_;
  std::array<std::vector<const char *>, 3> decoder_output_ptrs_;
  std::array<int32_t, 3> dec_chunk_begin_{};
  std::array<int32_t, 3> dec_chunk_layers_{};

  int32_t enc_layers_ = 0;
  int32_t enc_slots_ = 0;
  int32_t enc_heads_ = 0;
  int32_t enc_head_dim_ = 0;
  int32_t enc_keep_ = 0;
  int32_t dec_layers_ = 0;
  int32_t dec_slots_ = 0;
  int32_t dec_heads_ = 0;
  int32_t dec_head_dim_ = 0;
  int32_t dec_hidden_ = 0;
  int32_t audio_hidden_ = 0;
  int32_t vocab_size_ = 0;
};

OnlineAudio8AsrInfiniteModel::OnlineAudio8AsrInfiniteModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineAudio8AsrInfiniteModel::OnlineAudio8AsrInfiniteModel(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineAudio8AsrInfiniteModel::~OnlineAudio8AsrInfiniteModel() = default;

Ort::Value OnlineAudio8AsrInfiniteModel::ForwardTower(
    const float *pcm, int32_t n_pcm, int32_t num_tokens, int64_t pos_enc,
    std::vector<Ort::Value> *enc_k, std::vector<Ort::Value> *enc_v) const {
  return impl_->ForwardTower(pcm, n_pcm, num_tokens, pos_enc, enc_k, enc_v);
}

Ort::Value OnlineAudio8AsrInfiniteModel::ForwardDecoder(
    const int64_t *input_ids, int32_t num_tokens,
    const Ort::Value &audio_embeds, int64_t pos_dec,
    std::vector<Ort::Value> *dec_k, std::vector<Ort::Value> *dec_v) const {
  return impl_->ForwardDecoder(input_ids, num_tokens, audio_embeds, pos_dec,
                               dec_k, dec_v);
}

std::vector<Ort::Value> OnlineAudio8AsrInfiniteModel::CreateEmptyEncCache(
    int32_t which) const {
  return impl_->CreateEmptyEncCache(which);
}

std::vector<Ort::Value> OnlineAudio8AsrInfiniteModel::CreateEmptyDecCache(
    int32_t which) const {
  return impl_->CreateEmptyDecCache(which);
}

int64_t OnlineAudio8AsrInfiniteModel::EncRollShift(int64_t pos_enc,
                                                   int32_t num_frames) const {
  return impl_->EncRollShift(pos_enc, num_frames);
}

int64_t OnlineAudio8AsrInfiniteModel::DecRollShift(int64_t pos_dec,
                                                   int32_t num_tokens) const {
  return impl_->DecRollShift(pos_dec, num_tokens);
}

void OnlineAudio8AsrInfiniteModel::RollEncCache(
    int64_t shift, std::vector<Ort::Value> *enc_k,
    std::vector<Ort::Value> *enc_v) const {
  impl_->RollEncCache(shift, enc_k, enc_v);
}

void OnlineAudio8AsrInfiniteModel::RollDecCache(
    int64_t shift, std::vector<Ort::Value> *dec_k,
    std::vector<Ort::Value> *dec_v) const {
  impl_->RollDecCache(shift, dec_k, dec_v);
}

int32_t OnlineAudio8AsrInfiniteModel::EncSlots() const {
  return impl_->EncSlots();
}

int32_t OnlineAudio8AsrInfiniteModel::DecSlots() const {
  return impl_->DecSlots();
}

int32_t OnlineAudio8AsrInfiniteModel::NumEncLayers() const {
  return impl_->NumEncLayers();
}

int32_t OnlineAudio8AsrInfiniteModel::NumDecLayers() const {
  return impl_->NumDecLayers();
}

int32_t OnlineAudio8AsrInfiniteModel::EncHeads() const {
  return impl_->EncHeads();
}

int32_t OnlineAudio8AsrInfiniteModel::EncHeadDim() const {
  return impl_->EncHeadDim();
}

int32_t OnlineAudio8AsrInfiniteModel::DecKvHeads() const {
  return impl_->DecKvHeads();
}

int32_t OnlineAudio8AsrInfiniteModel::DecHeadDim() const {
  return impl_->DecHeadDim();
}

int32_t OnlineAudio8AsrInfiniteModel::DecHiddenSize() const {
  return impl_->DecHiddenSize();
}

int32_t OnlineAudio8AsrInfiniteModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineAudio8AsrInfiniteModel::EncRollKeep() const {
  return impl_->EncRollKeep();
}

const Audio8InfiniteGear &OnlineAudio8AsrInfiniteModel::Gear() const {
  return impl_->Gear();
}

OrtAllocator *OnlineAudio8AsrInfiniteModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OnlineAudio8AsrInfiniteModel::OnlineAudio8AsrInfiniteModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineAudio8AsrInfiniteModel::OnlineAudio8AsrInfiniteModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
