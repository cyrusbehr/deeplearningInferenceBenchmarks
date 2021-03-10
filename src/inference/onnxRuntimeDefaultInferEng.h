#pragma once
#include <memory>

#include "inferenceEngineTemplate.h"
#include "onnxruntime_cxx_api.h"

// Sample code here: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp#L24-L30

class InferenceEng : InferenceEngineTemplate {
public:
    InferenceEng(const std::string& modelDir);
    ~InferenceEng() = default;
    void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) override;
private:
    const std::string MODEL_NAME = "tfv5_reshaped_relu_wo_id.onnx";
    std::unique_ptr<Ort::Session> m_sessionPtr;
    std::unique_ptr<Ort::Env> m_envPtr;
    Ort::SessionOptions m_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> m_nodeNames;
    std::vector<int64_t> m_inputNodeDims;
};