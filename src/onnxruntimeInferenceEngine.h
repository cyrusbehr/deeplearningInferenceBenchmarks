#ifdef USE_ONNX

#pragma once

#include "onnxruntime_cxx_api.h"
#include "inferenceEngine.h"

class OnnxruntimeInferenceEngine
        : public EvalInferenceEngine {
public:
    explicit OnnxruntimeInferenceEngine(const std::string& modelPath);
    ~OnnxruntimeInferenceEngine() = default;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    std::unique_ptr<Ort::Session> m_sessionPtr;
    std::unique_ptr<Ort::Env> m_envPtr;
    Ort::SessionOptions m_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> m_nodeNames;
    std::vector<int64_t> m_inputNodeDims;

    char m_ompNumThreads[18];
};
#endif