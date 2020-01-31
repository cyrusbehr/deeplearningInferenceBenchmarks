#ifdef USE_ONNX

#include "onnxruntimeInferenceEngine.h"

OnnxruntimeInferenceEngine::OnnxruntimeInferenceEngine(const std::string& modelPath)
        : m_ompNumThreads("OMP_NUM_THREADS=1") {

    putenv(m_ompNumThreads);

    m_envPtr = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    m_options.SetIntraOpNumThreads(1);
    m_options.SetInterOpNumThreads(1);

    m_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    m_sessionPtr = std::make_unique<Ort::Session>(*m_envPtr, modelPath.c_str(), m_options);

    size_t numInputNodes = m_sessionPtr->GetInputCount();
    m_nodeNames.resize(numInputNodes);

    for (int i = 0; i < numInputNodes; i++) {
        char* input_name = m_sessionPtr->GetInputName(i, allocator);
        m_nodeNames[i] = input_name;

        Ort::TypeInfo typeInfo = m_sessionPtr->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        m_inputNodeDims = tensorInfo.GetShape();
    }
}

bool OnnxruntimeInferenceEngine::runInference(const cv::Mat& imgRGB) {
    size_t input_tensor_size = 112 * 112 * 3;

    std::vector<const char*> output_node_names = {"fc1"};

    cv::Mat imgRGBFloat;
    imgRGB.convertTo(imgRGBFloat, CV_32F);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(imgRGBFloat.data), input_tensor_size, m_inputNodeDims.data(), 4);
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors = m_sessionPtr->Run(Ort::RunOptions{nullptr}, m_nodeNames.data(), &input_tensor, 1, output_node_names.data(), 1);

    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    auto* floatarr = output_tensors.front().GetTensorMutableData<float>();
    return true;
}

#endif