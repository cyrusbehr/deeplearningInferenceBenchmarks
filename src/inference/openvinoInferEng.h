#pragma once

#include "inferenceEngineTemplate.h"
#include <inference_engine.hpp>

class InferenceEng: InferenceEngineTemplate {
public:
    InferenceEng(const std::string& modelDir);
    void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) override;
private:
    InferenceEngine::InferRequest m_inferenceRequest;
    std::string m_inputName;
    std::string m_outputName;
    const std::string XML_NAME = "openvino_rgb.xml";
};