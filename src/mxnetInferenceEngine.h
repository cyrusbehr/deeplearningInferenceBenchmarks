#ifdef USE_MXNET
#pragma once

#include "mxnet-cpp/MxNetCpp.h"
using namespace mxnet::cpp;

#include "inferenceEngine.h"


class MxnetInferenceEngine
        : public EvalInferenceEngine {
public:
    MxnetInferenceEngine(const std::string& binFilepath, const std::string& paramFilepath);
    ~MxnetInferenceEngine() override;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    NDArray mtxToNDArr(const cv::Mat& frame);

    std::map<std::string, mxnet::cpp::NDArray> m_argsMap;
    std::map<std::string, mxnet::cpp::NDArray> m_auxMap;
    mxnet::cpp::Context m_globalCtx;
    mxnet::cpp::Executor *m_executor = nullptr;
    mxnet::cpp::Symbol m_net;
};

#endif