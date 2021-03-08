#include <vector>
#include "onnxRuntimeDefaultInferEng.h"

InferenceEngine::InferenceEngine(const std::string &modelDir) {
    std::cout << "Using onnxruntime inference engine" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    m_envPtr = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    m_options.SetInterOpNumThreads(1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    m_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::string modelPath = modelDir + MODEL_NAME;
    m_sessionPtr = std::make_unique<Ort::Session>(*m_envPtr, modelPath.c_str(), m_options);
}

void InferenceEngine::runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) {
    cv::Mat imgRgbFloat;
    rgbImage.convertTo(imgRgbFloat, CV_32F, 1.0 / 255);

    size_t inputTensorSize = 112 * 112 * 3;

    std::vector<const char*> outputNodeNames = {"fc1"};
    std::vector<const char*> inputNodeNames = {"data"};
    std::vector<int64_t> inputNodeDims = {1, 3, 112, 112};

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(imgRgbFloat.data), inputTensorSize, inputNodeDims.data(), inputNodeDims.size());
    if (!input_tensor.IsTensor()) {
        throw std::runtime_error("Input is not a tensor");
    }

    // score model & input tensor, get back output tensor
    auto outputTensor = m_sessionPtr->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), 1);
    if (outputTensor.size() != 1 || !outputTensor.front().IsTensor()) {
        throw std::runtime_error("Error is not a tensor");
    }

    // Get pointer to output tensor float values
    float* floatarr = outputTensor.front().GetTensorMutableData<float>();

    for (int i = 0; i < 500; ++i) {
        output[i] = floatarr[i];
    }
}