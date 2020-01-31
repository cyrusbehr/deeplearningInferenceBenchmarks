#ifdef USE_TVM

#include <fstream>
#include "tvmInferenceEngine.h"

TvmInferenceEngine::TvmInferenceEngine(const std::string& paramsFilepath, const std::string& jsonFilepath, const std::string& libPath)
        : m_ompNumThreads("OMP_NUM_THREADS=1") {
    putenv(m_ompNumThreads);

    tvm::runtime::Module modSyslib = tvm::runtime::Module::LoadFromFile(libPath);

    std::ifstream jsonInput(jsonFilepath);
    const std::string jsonData((std::istreambuf_iterator<char>(jsonInput)), std::istreambuf_iterator<char>());
    jsonInput.close();

    // Get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(jsonData, modSyslib, DEVICE_TYPE, DEVICE_ID);
    m_handle = std::make_unique<tvm::runtime::Module>(mod);

    // Load param
    std::ifstream paramsInput(paramsFilepath, std::ios::binary);
    const std::string paramsData((std::istreambuf_iterator<char>(paramsInput)), std::istreambuf_iterator<char>());
    paramsInput.close();

    TVMByteArray paramsArr;
    paramsArr.data = paramsData.c_str();
    paramsArr.size = paramsData.length();

    tvm::runtime::PackedFunc loadParamsFunc = mod.GetFunction("load_params");
    loadParamsFunc(paramsArr);
}

TvmInferenceEngine::~ TvmInferenceEngine() = default;

bool TvmInferenceEngine::runInference(const cv::Mat& imgRGB) {
    cv::Mat tensor = cv::dnn::blobFromImage(imgRGB, 1.0, cv::Size(IMG_WIDTH, IMG_HEIGH), cv::Scalar(0, 0, 0), true);
    DLTensor* input;

    const int64_t inShape[IN_NDIMS] = {BATCH_SIZE, CHANNELS, IMG_WIDTH, IMG_HEIGH};
    TVMArrayAlloc(inShape, IN_NDIMS, DTYPE_CODE, DTYPE_BITS, DTYPE_LANES, DEVICE_TYPE, DEVICE_ID, &input);
    TVMArrayCopyFromBytes(input, tensor.data, IMG_WIDTH * IMG_HEIGH * CHANNELS * sizeof(float));

    tvm::runtime::PackedFunc setInputFunc = m_handle->GetFunction("set_input");
    setInputFunc("data", input);
    tvm::runtime::PackedFunc run = m_handle->GetFunction("run");
    run();

    tvm::runtime::PackedFunc getOutput = m_handle->GetFunction("get_output");
    tvm::runtime::NDArray res = getOutput(0);

    cv::Mat output(FEATURE_SIZE, 1, CV_32F);
    memcpy(output.data,res->data, FEATURE_SIZE * sizeof(float));

    TVMArrayFree(input);

    return true;
}

#endif