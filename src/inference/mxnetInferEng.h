#pragma once

#include "inferenceEngineTemplate.h"
#include "mxnet-cpp/MxNetCpp.h"
using namespace mxnet::cpp;

class InferenceEng : InferenceEngineTemplate {
public:
    InferenceEng(const std::string& modelDir);
    ~InferenceEng();
    void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) override;
private:
    inline NDArray matToNDArray(cv::Mat rgb_image, Context ctx = Context::cpu());
    std::map<std::string, mxnet::cpp::NDArray> m_argsMap;
    std::map<std::string, mxnet::cpp::NDArray> m_auxMap;
    mxnet::cpp::Context m_globalCtx;
    mxnet::cpp::Executor *m_executor = nullptr;
    mxnet::cpp::Symbol m_net;

    const std::string JSON_NAME = "mxnet.json";
    const std::string PARAMS_NAME = "mxnet.params";
};

