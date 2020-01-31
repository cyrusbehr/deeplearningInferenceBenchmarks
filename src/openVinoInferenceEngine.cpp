#ifdef USE_OPENVINO
#include "openVinoInferenceEngine.h"

using namespace InferenceEngine;

OpenVinoInferenceEngine::OpenVinoInferenceEngine(const std::string& xmlFilepath, const std::string& binFilepath) {
    m_networkReader.ReadNetwork(fileNameToString(xmlFilepath));
    m_networkReader.ReadWeights(fileNameToString(binFilepath));
    m_networkReader.getNetwork().setBatchSize(1);
    m_network = m_networkReader.getNetwork();

    InputInfo::Ptr input_info = m_network.getInputsInfo().begin()->second;
    m_inputName = m_network.getInputsInfo().begin()->first;
    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);

    DataPtr output_info = m_network.getOutputsInfo().begin()->second;
    m_outputName = m_network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);

    std::map< std::string, std::string > options;
    options[PluginConfigParams::KEY_CPU_BIND_THREAD] = PluginConfigParams::NO;
    options[PluginConfigParams::KEY_CPU_THREADS_NUM] = "1";

    m_executableNetwork = m_inferenceEngine.LoadNetwork(m_network, DEVICE_NAME, options);

}

bool OpenVinoInferenceEngine::runInference(const cv::Mat& imgRGB) {
    InferRequest infer_request = m_executableNetwork.CreateInferRequest();
    Blob::Ptr imgBlob = wrapMat2Blob(imgRGB);
    infer_request.SetBlob(m_inputName, imgBlob);

    infer_request.Infer();

    Blob::Ptr output = infer_request.GetBlob(m_outputName);

    return true;
}
#endif