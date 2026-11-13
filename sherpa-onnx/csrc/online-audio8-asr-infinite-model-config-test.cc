// sherpa-onnx/csrc/online-audio8-asr-infinite-model-config-test.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

// The tests below only cover the pure bookkeeping of Audio8-ASR-Infinite:
// the gear geometry, the window schedule it implies and the config validation.
// They never load an ONNX model, so they are fast enough to run on every PR.
//
// The `max_new_tokens` safety cap of
// `OnlineRecognizerAudio8AsrInfiniteImpl::IsReady()` needs a loaded model (it
// compares against the per-stream state), so it is exercised by the end-to-end
// runs of the example instead of by this file.

namespace sherpa_onnx {

namespace {

struct Audio8InfiniteTestDir {
  std::filesystem::path dir;

  static void Touch(const std::filesystem::path &p) {
    std::ofstream ofs(p);
    ofs << "x";
  }

  void SetUp() {
    std::filesystem::path base = std::filesystem::temp_directory_path();
    dir = base / "sherpa-onnx-audio8-asr-infinite-model-config-test";
    std::filesystem::remove_all(dir);

    std::filesystem::create_directories(dir / "tokenizer");
    std::filesystem::create_directories(dir / "test_wavs");

    for (const char *name : {"tower.onnx", "decoder_0.onnx", "decoder_1.onnx",
                             "decoder_2.onnx", "frontend_trim0.onnx",
                             "frontend_trim2.onnx"}) {
      Touch(dir / name);
    }

    for (const char *name :
         {"vocab.json", "merges.txt", "tokenizer_config.json"}) {
      Touch(dir / "tokenizer" / name);
    }
  }

  void TearDown() const { std::filesystem::remove_all(dir); }

  std::string Path() const { return dir.string(); }

  std::string Tokenizer() const { return (dir / "tokenizer").string(); }
};

class Audio8InfiniteModelConfigTest : public ::testing::Test {
 protected:
  void SetUp() override { env_.SetUp(); }
  void TearDown() override { env_.TearDown(); }

  // A config that passes Validate() out of the box.
  OnlineAudio8AsrInfiniteModelConfig GoodConfig() const {
    OnlineAudio8AsrInfiniteModelConfig c;
    c.model_dir = env_.Path();
    c.tokenizer = env_.Tokenizer();
    return c;
  }

  Audio8InfiniteTestDir env_;
};

// 16 kHz * clock_ms / 1000.
int32_t SamplesPerToken(int32_t clock_ms) { return 16000 * clock_ms / 1000; }

// Number of PCM samples of the window that produces `num_tokens` tokens with
// the look-back / look-ahead of the reference schedule.
int32_t WindowSamples(const Audio8InfiniteGear &gear, int32_t num_tokens) {
  return num_tokens * gear.samples_per_token +
         Audio8InfiniteGear::kLookBackSamples +
         Audio8InfiniteGear::kLookAheadSamples;
}

// The first window of a segment starts at virtual position 0, so its look-back
// is clamped to 0 and only the look-ahead is added.
int32_t FirstWindowSamples(const Audio8InfiniteGear &gear) {
  return gear.prefill_tokens * gear.samples_per_token +
         Audio8InfiniteGear::kLookAheadSamples;
}

}  // namespace

TEST(Audio8InfiniteGearTest, ClockGeometry160Ms) {
  Audio8InfiniteGear gear;
  std::string error;
  ASSERT_TRUE(Audio8InfiniteGear::FromConfig(8, 480, &gear, &error)) << error;

  EXPECT_EQ(gear.frame_len, 8);
  EXPECT_EQ(gear.clock_ms, 160);
  EXPECT_EQ(gear.delay_tokens, 3);
  EXPECT_EQ(gear.left_pad_tokens, 9);
  EXPECT_EQ(gear.prefill_tokens, 13);
  EXPECT_EQ(gear.right_pad_tokens, 14);
  EXPECT_EQ(gear.samples_per_token, 2560);
  EXPECT_EQ(gear.left_pad_samples, 23040);
}

TEST(Audio8InfiniteGearTest, ClockGeometry80MsAnd120Ms) {
  for (const auto &p : {std::pair<int32_t, int32_t>{4, 80}, {6, 120}}) {
    const int32_t frame_len = p.first;
    const int32_t clock_ms = p.second;

    Audio8InfiniteGear gear;
    std::string error;
    ASSERT_TRUE(Audio8InfiniteGear::FromConfig(frame_len, 480, &gear, &error))
        << error;

    EXPECT_EQ(gear.frame_len, frame_len);
    EXPECT_EQ(gear.clock_ms, clock_ms);
    // The delay is given in milliseconds, so the number of tokens depends on
    // the clock: 480 ms is 6 tokens at 80 ms and 4 tokens at 120 ms.
    EXPECT_EQ(gear.delay_tokens, 480 / clock_ms);
    EXPECT_EQ(gear.samples_per_token, SamplesPerToken(clock_ms));
    // Every supported gear starts a segment with the same 1.44 s of silence.
    EXPECT_EQ(gear.left_pad_samples, 23040);
    EXPECT_EQ(gear.left_pad_tokens, 23040 / gear.samples_per_token);
    // prefill = left pad + delay + 1 (the last clock).
    EXPECT_EQ(gear.prefill_tokens,
              gear.left_pad_tokens + gear.delay_tokens + 1);
    // right pad = delay + 1 + 10 text tokens of the reference schedule.
    EXPECT_EQ(gear.right_pad_tokens, gear.delay_tokens + 11);
  }
}

TEST(Audio8InfiniteGearTest, DelayIsQuantizedToTheClock) {
  Audio8InfiniteGear gear;
  std::string error;

  // 960 ms = 6 tokens at 160 ms, 12 tokens at 80 ms.
  ASSERT_TRUE(Audio8InfiniteGear::FromConfig(8, 960, &gear, &error)) << error;
  EXPECT_EQ(gear.delay_tokens, 6);
  EXPECT_EQ(gear.prefill_tokens, 9 + 6 + 1);

  ASSERT_TRUE(Audio8InfiniteGear::FromConfig(4, 960, &gear, &error)) << error;
  EXPECT_EQ(gear.delay_tokens, 12);
  EXPECT_EQ(gear.prefill_tokens, 18 + 12 + 1);

  // 160 ms = one token at 160 ms.
  ASSERT_TRUE(Audio8InfiniteGear::FromConfig(8, 160, &gear, &error)) << error;
  EXPECT_EQ(gear.delay_tokens, 1);
  EXPECT_EQ(gear.prefill_tokens, 11);
}

TEST(Audio8InfiniteGearTest, RejectsUnsupportedFrameLength) {
  for (int32_t frame_len : {0, 1, 2, 3, 5, 7, 9, 10, -4}) {
    Audio8InfiniteGear gear;
    std::string error;
    EXPECT_FALSE(Audio8InfiniteGear::FromConfig(frame_len, 480, &gear, &error))
        << "frame_len " << frame_len << " should be rejected";
    EXPECT_NE(error.find("frame-len"), std::string::npos) << error;
  }
}

TEST(Audio8InfiniteGearTest, RejectsDelayThatIsNotAMultipleOfTheClock) {
  // clock 160 ms: 500 and 1 are not multiples of it.
  for (int32_t delay_ms : {1, 40, 100, 500, 481}) {
    Audio8InfiniteGear gear;
    std::string error;
    EXPECT_FALSE(Audio8InfiniteGear::FromConfig(8, delay_ms, &gear, &error))
        << "delay_ms " << delay_ms << " should be rejected at 160 ms";
    EXPECT_NE(error.find("delay-ms"), std::string::npos) << error;
  }

  for (int32_t delay_ms : {0, -160}) {
    Audio8InfiniteGear gear;
    std::string error;
    EXPECT_FALSE(Audio8InfiniteGear::FromConfig(8, delay_ms, &gear, &error));
  }

  // 120 ms clock: 480 is a multiple, 500 is not.
  {
    Audio8InfiniteGear gear;
    std::string error;
    EXPECT_TRUE(Audio8InfiniteGear::FromConfig(6, 480, &gear, &error)) << error;
    EXPECT_FALSE(Audio8InfiniteGear::FromConfig(6, 500, &gear, &error));
  }
}

TEST(Audio8InfiniteGearTest, TrimDerivationOfFirstAndLaterWindows) {
  // The first window of a segment feeds `prefill_tokens` tokens and the
  // frontend must drop 0 frames; every later window feeds one token and must
  // drop 2. If this does not hold, the gear does not match the export.
  for (int32_t frame_len : {4, 6, 8}) {
    Audio8InfiniteGear gear;
    std::string error;
    ASSERT_TRUE(Audio8InfiniteGear::FromConfig(frame_len, 480, &gear, &error))
        << error;

    // The first window of a segment needs trim 0.
    const int32_t first_window = FirstWindowSamples(gear);
    EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(first_window),
              gear.prefill_tokens * gear.frame_len)
        << "frame_len " << frame_len;
    EXPECT_EQ(gear.TrimForWindow(first_window, gear.prefill_tokens), 0)
        << "frame_len " << frame_len;

    // Every later window feeds one token and drops 2 frames, for every gear.
    const int32_t later_window = WindowSamples(gear, 1);
    EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(later_window),
              gear.frame_len + 2);
    EXPECT_EQ(gear.TrimForWindow(later_window, 1), 2);
  }
}

TEST(Audio8InfiniteGearTest, MelAndTowerFrameCounts) {
  EXPECT_EQ(Audio8InfiniteGear::MelFrames(0), 0);
  EXPECT_EQ(Audio8InfiniteGear::MelFrames(160), 1);
  EXPECT_EQ(Audio8InfiniteGear::MelFrames(160 * 100), 100);
  // Not a multiple of 10 ms: integer division, like the frontend graph.
  EXPECT_EQ(Audio8InfiniteGear::MelFrames(33320), 208);

  // conv1 (k=3, s=1) keeps the length, conv2 (k=3, s=2) halves it.
  EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(0), 0);
  EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(160), 0);
  EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(160 * 3), 1);
  EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(33320), 104);
  EXPECT_EQ(Audio8InfiniteGear::TowerFramesBeforeTrim(3440), 10);
}

TEST_F(Audio8InfiniteModelConfigTest, DefaultsMatchTheCheckpoint) {
  OnlineAudio8AsrInfiniteModelConfig c;

  EXPECT_FALSE(c.IsSet());
  EXPECT_TRUE(c.model_dir.empty());
  EXPECT_EQ(c.frame_len, 8);
  EXPECT_EQ(c.delay_ms, 480);
  EXPECT_EQ(c.max_total_len, 512);
  EXPECT_EQ(c.language, "zh");
  EXPECT_EQ(c.enc_window, 750);
  EXPECT_EQ(c.max_new_tokens, 0);
  EXPECT_FALSE(c.debug);
}

TEST_F(Audio8InfiniteModelConfigTest, IsSetFollowsTheFamilyDispatch) {
  OnlineAudio8AsrInfiniteModelConfig c;
  EXPECT_FALSE(c.IsSet());

  c.model_dir = env_.Path();
  EXPECT_TRUE(c.IsSet());

  OnlineAudio8AsrInfiniteModelConfig c2;
  c2.encoder = env_.Path() + "/tower.onnx";
  EXPECT_TRUE(c2.IsSet());

  OnlineAudio8AsrInfiniteModelConfig c3;
  c3.decoder2 = env_.Path() + "/decoder_2.onnx";
  EXPECT_TRUE(c3.IsSet());
}

TEST_F(Audio8InfiniteModelConfigTest, ResolveUsesStandardFileNames) {
  OnlineAudio8AsrInfiniteModelConfig c;
  EXPECT_EQ(c.ResolveEncoder(), "tower.onnx");
  EXPECT_EQ(c.ResolveDecoder(0), "decoder_0.onnx");
  EXPECT_EQ(c.ResolveDecoder(2), "decoder_2.onnx");
  EXPECT_EQ(c.ResolveFrontend(0), "frontend_trim0.onnx");
  EXPECT_EQ(c.ResolveFrontend(2), "frontend_trim2.onnx");
  // Only trim 0 and trim 2 are part of the schedule.
  EXPECT_TRUE(c.ResolveDecoder(3).empty());
  EXPECT_TRUE(c.ResolveDecoder(-1).empty());
  EXPECT_TRUE(c.ResolveFrontend(1).empty());

  c.model_dir = env_.Path();
  EXPECT_EQ(c.ResolveEncoder(), env_.Path() + "/tower.onnx");
  EXPECT_EQ(c.ResolveFrontend(2), env_.Path() + "/frontend_trim2.onnx");

  c.model_dir = env_.Path() + "/";
  EXPECT_EQ(c.ResolveEncoder(), env_.Path() + "/tower.onnx");

  // An explicit path always wins over model_dir.
  c.encoder = "/tmp/custom-tower.onnx";
  c.decoder1 = "/tmp/custom-decoder-1.onnx";
  c.frontend_trim0 = "/tmp/custom-frontend-0.onnx";
  EXPECT_EQ(c.ResolveEncoder(), "/tmp/custom-tower.onnx");
  EXPECT_EQ(c.ResolveDecoder(1), "/tmp/custom-decoder-1.onnx");
  EXPECT_EQ(c.ResolveFrontend(0), "/tmp/custom-frontend-0.onnx");
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateAcceptsACompleteBundle) {
  auto c = GoodConfig();
  EXPECT_TRUE(c.Validate());

  // Resolved against model_dir.
  c.encoder.clear();
  EXPECT_TRUE(c.Validate());

  // Both gears of the checkpoint are accepted by the config; the graphs have
  // to match it, which the model class checks at load time.
  c.frame_len = 4;
  EXPECT_TRUE(c.Validate());
  c.frame_len = 6;
  EXPECT_TRUE(c.Validate());
  c.frame_len = 8;

  c.language = "en";
  EXPECT_TRUE(c.Validate());
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateRejectsMissingModelDir) {
  OnlineAudio8AsrInfiniteModelConfig c;
  c.tokenizer = env_.Tokenizer();
  EXPECT_FALSE(c.Validate());

  // An encoder alone is enough to dispatch to this family, but the file has
  // to exist.
  c.encoder = env_.Path() + "/no-such-tower.onnx";
  EXPECT_TRUE(c.IsSet());
  EXPECT_FALSE(c.Validate());
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateRejectsMissingGraphs) {
  for (const char *name : {"tower.onnx", "decoder_0.onnx", "decoder_1.onnx",
                           "decoder_2.onnx", "frontend_trim0.onnx",
                           "frontend_trim2.onnx"}) {
    // Every test gets a fresh directory from SetUp().
    std::filesystem::remove(env_.dir / name);
    auto c = GoodConfig();
    EXPECT_FALSE(c.Validate()) << name << " should be required";
    Audio8InfiniteTestDir::Touch(env_.dir / name);
  }
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateRejectsMissingTokenizer) {
  auto c = GoodConfig();
  c.tokenizer = "";
  EXPECT_FALSE(c.Validate());

  c.tokenizer = env_.Path() + "/no-such-tokenizer";
  EXPECT_FALSE(c.Validate());

  // vocab.json and merges.txt can be derived from the checkpoint's
  // tokenizer.json, but they have to be there.
  for (const char *name :
       {"vocab.json", "merges.txt", "tokenizer_config.json"}) {
    std::filesystem::remove(std::filesystem::path(env_.Tokenizer()) / name);
    auto c2 = GoodConfig();
    EXPECT_FALSE(c2.Validate()) << name << " should be required";
    Audio8InfiniteTestDir::Touch(std::filesystem::path(env_.Tokenizer()) /
                                 name);
  }
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateRejectsBadGear) {
  auto c = GoodConfig();

  c.frame_len = 5;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.delay_ms = 500;  // not a multiple of 160
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.frame_len = 4;
  c.delay_ms = 500;  // not a multiple of 80 either
  EXPECT_FALSE(c.Validate());
}

TEST_F(Audio8InfiniteModelConfigTest, ValidateRejectsBadLimits) {
  auto c = GoodConfig();

  c.max_total_len = 0;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.max_total_len = -1;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.enc_window = 0;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.enc_window = -750;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.max_new_tokens = -1;
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.language = "fr";
  EXPECT_FALSE(c.Validate());

  c = GoodConfig();
  c.language = "";
  EXPECT_FALSE(c.Validate());
}

TEST_F(Audio8InfiniteModelConfigTest, ToStringMentionsEveryField) {
  auto c = GoodConfig();
  c.frame_len = 4;
  c.delay_ms = 960;
  c.language = "en";
  c.max_new_tokens = 7;
  c.debug = true;

  const std::string s = c.ToString();
  EXPECT_NE(s.find("OnlineAudio8AsrInfiniteModelConfig"), std::string::npos);
  EXPECT_NE(s.find("frame_len=4"), std::string::npos) << s;
  EXPECT_NE(s.find("delay_ms=960"), std::string::npos) << s;
  EXPECT_NE(s.find("max_total_len=512"), std::string::npos) << s;
  EXPECT_NE(s.find("language=\"en\""), std::string::npos) << s;
  EXPECT_NE(s.find("enc_window=750"), std::string::npos) << s;
  EXPECT_NE(s.find("max_new_tokens=7"), std::string::npos) << s;
  EXPECT_NE(s.find("frontend_trim2"), std::string::npos) << s;
  EXPECT_NE(s.find("decoder2"), std::string::npos) << s;
}

}  // namespace sherpa_onnx
