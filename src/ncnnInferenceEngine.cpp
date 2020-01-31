#include "ncnnInferenceEngine.h"

NcnnInferenceEngine::NcnnInferenceEngine(const std::string& binFilepath, const std::string& paramFilepath) {
    m_net.load_param(paramFilepath.c_str());
    m_net.load_model(binFilepath.c_str());
}

bool NcnnInferenceEngine::runInference(const cv::Mat& imgRGB) {
    ncnn::Mat in = ncnn::Mat::from_pixels(imgRGB.data, ncnn::Mat::PIXEL_RGB, imgRGB.cols, imgRGB.rows);
    ncnn::Extractor ex = m_net.create_extractor();
    ex.set_num_threads(1);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);

    return out.data != nullptr;
}

NcnnInferenceEngine::~NcnnInferenceEngine()= default;;
