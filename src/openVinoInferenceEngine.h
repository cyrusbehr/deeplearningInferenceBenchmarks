#ifdef USE_OPENVINO

#include <string>

#pragma once
#include "inferenceEngine.h"
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/common.hpp>

class OpenVinoInferenceEngine
        : public EvalInferenceEngine {
public:
    OpenVinoInferenceEngine(const std::string& xmlFilepath, const std::string& binFilepath);
    ~OpenVinoInferenceEngine() override = default;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    InferenceEngine::Core m_inferenceEngine;
    InferenceEngine::CNNNetReader m_networkReader;
    InferenceEngine::CNNNetwork m_network;
    InferenceEngine::ExecutableNetwork m_executableNetwork;

    std::string m_inputName;
    std::string m_outputName;

    const std::string DEVICE_NAME = "CPU";
};
#endif