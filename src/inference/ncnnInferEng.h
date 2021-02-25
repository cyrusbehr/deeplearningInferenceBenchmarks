#pragma once

#include "inferenceEngineTemplate.h"
#include "net.h"

class InferenceEngine : InferenceEngineTemplate {
public:
    InferenceEngine(const std::string& modelDir);
    ~InferenceEngine() = default;
    void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) override;
private:
    ncnn::Net m_net;
    int m_numThreads = -1;

    const std::string WEIGHTS_NAME = "ncnn_tfv5.bin";
    const std::string PARAMS_NAME = "ncnn_tfv5.param";
};