#include "openvinoInferEng.h"
#include "./openvino/common.hpp"

using namespace InferenceEngine;

static UNUSED InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
            strideW == channels &&
            strideH == channels * width;

    if (!is_dense) THROW_IE_EXCEPTION
                << "Doesn't support conversion from not dense cv::Mat";

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

InferenceEng::InferenceEng(const std::string &modelDir) {
    std::cout << "Using openvino inference engine" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    const std::string xmlPath = modelDir + "/" + XML_NAME;

    Core ie;
    CNNNetwork network = ie.ReadNetwork(xmlPath);

    InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
    m_inputName = network.getInputsInfo().begin()->first;
    inputInfo->setPrecision(Precision::U8);
    inputInfo->setLayout(Layout::NHWC);

    DataPtr outputInfo = network.getOutputsInfo().begin()->second;
    m_outputName = network.getOutputsInfo().begin()->first;
    outputInfo->setPrecision(Precision::FP32);
    std::map< std::string, std::string > options;

    // Read number of threads to use
    if(const char* envP = std::getenv("OMP_NUM_THREADS")) {
        // https://docs.openvinotoolkit.org/cn/2021.1/namespaceInferenceEngine_1_1PluginConfigParams.html#a1264fc1aa7f58c908e884eb8fbaff8b2
        std::stringstream strValue;
        strValue << envP;
        options[PluginConfigParams::KEY_CPU_BIND_THREAD] = PluginConfigParams::NO;
        options[PluginConfigParams::KEY_CPU_THREADS_NUM] = strValue.str();
    }

    ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", options);
    m_inferenceRequest = executable_network.CreateInferRequest();
}

void InferenceEng::runInference(const cv::Mat &rgbImage, std::array<float, 500> &output) {
    // During model optimization, pass -reverse_input_channels flag so that input is rgb
    Blob::Ptr imgBlob = wrapMat2Blob(rgbImage);
    m_inferenceRequest.SetBlob(m_inputName, imgBlob);

    m_inferenceRequest.Infer();
    Blob::Ptr outputBlob = m_inferenceRequest.GetBlob(m_outputName);
    const int dims = outputBlob->size();
    memcpy(output.data(), outputBlob->buffer(), dims * sizeof(float));
}