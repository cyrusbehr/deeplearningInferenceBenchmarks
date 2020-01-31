#pragma once
#include "inferenceEngine.h"

#include "net.h"

class NcnnInferenceEngine
        : public EvalInferenceEngine {
public:
    NcnnInferenceEngine(const std::string& binFilepath, const std::string& paramFilepath);
    ~NcnnInferenceEngine() override;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    ncnn::Net m_net;
};
