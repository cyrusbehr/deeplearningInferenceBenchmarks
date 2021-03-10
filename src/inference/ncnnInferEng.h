#pragma once

#include "inferenceEngineTemplate.h"
#include "net.h"

class InferenceEng : InferenceEngineTemplate {
public:
    InferenceEng(const std::string& modelDir);
    ~InferenceEng() = default;
    void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) override;
private:
    ncnn::Net m_net;
    int m_numThreads = -1;

    const std::string WEIGHTS_NAME = "ncnn.bin";
    const std::string PARAMS_NAME = "ncnn.param";
};