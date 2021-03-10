#pragma once

#include <memory>


#ifdef USE_MXNET
#include "mxnetInferEng.h"
#elif USE_ONNX_DEFAULT
#include "onnxRuntimeDefaultInferEng.h"
#elif USE_OPENVINO
#include "openvinoInferEng.h"
#else
#include "ncnnInferEng.h"
#endif

class InferenceManager {
public:
    InferenceManager(const std::string& modelDir);

    void runBenchmark(unsigned int numIterations = 1000);
private:
    std::unique_ptr<InferenceEng> m_inferenceEnginePtr = nullptr;
    const std::string m_imagePath;

    void readTemplateFromDisk(const std::string& templatePath, std::array<float, 500>& templ);
    float dotProduct(const std::array<float, 500>& v1, const std::array<float, 500>& v2);
    void normalize(std::array<float, 500>& v);
};