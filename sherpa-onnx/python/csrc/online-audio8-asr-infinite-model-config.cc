// sherpa-onnx/python/csrc/online-audio8-asr-infinite-model-config.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/python/csrc/online-audio8-asr-infinite-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/online-audio8-asr-infinite-model-config.h"

namespace sherpa_onnx {

void PybindOnlineAudio8AsrInfiniteModelConfig(py::module *m) {
  using PyClass = OnlineAudio8AsrInfiniteModelConfig;
  py::class_<PyClass>(*m, "OnlineAudio8AsrInfiniteModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t, int32_t,
                    int32_t, const std::string &, int32_t, int32_t, bool>(),
           py::arg("model_dir") = "", py::arg("encoder") = "",
           py::arg("decoder0") = "", py::arg("decoder1") = "",
           py::arg("decoder2") = "", py::arg("frontend_trim0") = "",
           py::arg("frontend_trim2") = "", py::arg("tokenizer") = "",
           py::arg("frame_len") = 8, py::arg("delay_ms") = 480,
           py::arg("max_total_len") = 512, py::arg("language") = "zh",
           py::arg("enc_window") = 750, py::arg("max_new_tokens") = 0,
           py::arg("debug") = false)
      .def_readwrite("model_dir", &PyClass::model_dir)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder0", &PyClass::decoder0)
      .def_readwrite("decoder1", &PyClass::decoder1)
      .def_readwrite("decoder2", &PyClass::decoder2)
      .def_readwrite("frontend_trim0", &PyClass::frontend_trim0)
      .def_readwrite("frontend_trim2", &PyClass::frontend_trim2)
      .def_readwrite("tokenizer", &PyClass::tokenizer)
      .def_readwrite("frame_len", &PyClass::frame_len)
      .def_readwrite("delay_ms", &PyClass::delay_ms)
      .def_readwrite("max_total_len", &PyClass::max_total_len)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("enc_window", &PyClass::enc_window)
      .def_readwrite("max_new_tokens", &PyClass::max_new_tokens)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
