#ifdef USE_TVM
#pragma once
#include "inferenceEngine.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/threading_backend.h"

class TvmInferenceEngine
        : public EvalInferenceEngine {
public:
    TvmInferenceEngine(const std::string& paramsFilepath, const std::string& jsonFilepath, const std::string& libPath);
    ~ TvmInferenceEngine() override;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    std::unique_ptr<tvm::runtime::Module> m_handle = nullptr;

    static constexpr int DEVICE_TYPE = kDLCPU;
    static constexpr int DEVICE_ID = 0;
    static constexpr int DTYPE_CODE = kDLFloat;
    static constexpr int DTYPE_BITS = 32;
    static constexpr int DTYPE_LANES = 1;
    static constexpr int IN_NDIMS = 4;

    char m_ompNumThreads[18];
};
#endif