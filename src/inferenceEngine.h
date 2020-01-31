#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

class EvalInferenceEngine {
public:
    enum class InferenceEngineType {
        MXNET,
        NCNN,
        TVM,
        TENSORFLOW,
        ONNXRUNTIME,
        OPENVINO
    };

    EvalInferenceEngine();
    virtual ~EvalInferenceEngine() = default;
    virtual bool runInference(const cv::Mat& imgRGB) = 0;

protected:
    static constexpr size_t FEATURE_SIZE = 512;
    static constexpr size_t IMG_WIDTH = 112;
    static constexpr size_t IMG_HEIGH = 112;
    static constexpr size_t BATCH_SIZE = 1;
    static constexpr size_t CHANNELS = 3;

private:
    char m_mxnetEngineType[30];
    char m_ompNumThreads[18];
};



