// c-api-examples/audio8-asr-infinite-c-api.c
//
// Copyright (c)  2026  yuyun2000

//
// This file demonstrates how to use Audio8-ASR-Infinite with sherpa-onnx's C
// API.
//
// Audio8-ASR-Infinite is a *streaming* (online) model, so this example feeds a
// wave file in 100 ms chunks, decodes after every chunk, and prints the
// partial transcript while the audio is still being fed.  The final text is
// printed once the input is finished.
//
// Audio8-ASR-Infinite is a token-clock model: it emits one text token for
// every `frame_len * 20 ms` of audio, so the first characters appear after
// roughly `(left_pad + delay) * clock_ms` and then grow steadily.
//
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-infinite-int8.tar.bz2
tar xvf sherpa-onnx-audio8-asr-infinite-int8.tar.bz2
rm sherpa-onnx-audio8-asr-infinite-int8.tar.bz2
*/
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

// Number of samples of one chunk: 1600 = 100 ms at 16 kHz.
#define CHUNK 1600

// Prints the part of `text` that has not been printed yet, so that the partial
// transcript grows on a single line while the audio is being fed.
static void PrintPartial(char **printed, const char *text) {
  size_t n_old = (*printed == NULL) ? 0 : strlen(*printed);
  size_t n_new = strlen(text);

  if (n_new <= n_old) {
    return;
  }

  if (n_old > 0 && strncmp(*printed, text, n_old) != 0) {
    // The text was rewritten from scratch (it can happen after a stream
    // reset).  Start a new line.
    printf("\n");
    n_old = 0;
  }

  fputs(text + n_old, stdout);
  fflush(stdout);

  *printed = (char *)realloc(*printed, n_new + 1);
  memcpy(*printed, text, n_new + 1);
}

int32_t main() {
  // clang-format off
  const char *model_dir =
      "./sherpa-onnx-audio8-asr-infinite-int8";
  const char *tokenizer_dir =
      "./sherpa-onnx-audio8-asr-infinite-int8/tokenizer";
  const char *wav_filename =
      "./sherpa-onnx-audio8-asr-infinite-int8/test_wavs/0.wav";
  // clang-format on

  const char *provider = "cpu";

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 128;

  // Leaving the per-graph paths empty resolves them against model_dir.
  config.model_config.audio8_asr_infinite.model_dir = model_dir;
  config.model_config.audio8_asr_infinite.tokenizer = tokenizer_dir;

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
  config.model_config.audio8_asr_infinite.debug = 0;

  // There is no tokens.txt: this model family loads its tokenizer from the
  // `tokenizer` directory.
  config.model_config.num_threads = 2;
  config.model_config.provider = provider;
  config.model_config.debug = 0;

  // Audio8-ASR-Infinite supports only greedy_search.
  config.decoding_method = "greedy_search";

  // The acoustic endpointing fallback cuts a sentence at every long pause, so
  // it is disabled here.  Set it to 1 if you want to reset the stream at the
  // end of an utterance.
  config.enable_endpoint = 0;

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read '%s'\n", wav_filename);
    return -1;
  }

  fprintf(stderr, "Loading model from '%s'\n", model_dir);
  fprintf(stderr, "This model is large, please wait\n");

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&config);
  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxCreateOnlineStream(recognizer);

  char *printed = NULL;
  int32_t offset = 0;

  while (offset < wave->num_samples) {
    int32_t n = CHUNK;
    if (offset + n > wave->num_samples) {
      n = wave->num_samples - offset;
    }

    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + offset, n);
    offset += n;

    // One decode step per clock.  While the stream is ready there is always a
    // full window of audio, so this never busy-waits.
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);

      const SherpaOnnxOnlineRecognizerResult *r =
          SherpaOnnxGetOnlineStreamResult(recognizer, stream);
      PrintPartial(&printed, r->text);
      SherpaOnnxDestroyOnlineRecognizerResult(r);
    }
  }

  SherpaOnnxFreeWave(wave);

  // No tail padding is needed: the model itself pads the audio with
  // `right_pad_tokens` clocks of silence once the input is finished.
  SherpaOnnxOnlineStreamInputFinished(stream);

  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);
    PrintPartial(&printed, r->text);
    SherpaOnnxDestroyOnlineRecognizerResult(r);
  }

  const SherpaOnnxOnlineRecognizerResult *result =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);

  printf("\n");
  printf("Final text: %s\n", result->text);

  free(printed);
  SherpaOnnxDestroyOnlineRecognizerResult(result);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);

  return 0;
}
