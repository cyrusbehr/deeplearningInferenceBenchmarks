#include "ncnnInferEng.h"

InferenceEng::InferenceEng(const std::string &modelDir) {
    std::cout << "Using ncnn inference engine" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    const std::string paramFilepath = modelDir + PARAMS_NAME;
    const std::string weighsFilepath = modelDir + WEIGHTS_NAME;
    m_net.load_param(paramFilepath.c_str());
    m_net.load_model(weighsFilepath.c_str());

    // Read number of threads to use
    if(const char* envP = std::getenv("OMP_NUM_THREADS")) {
        std::stringstream strValue;
        strValue << envP;
        strValue >> m_numThreads;
    }
}

void InferenceEng::runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) {
    auto in = ncnn::Mat::from_pixels(rgbImage.data, ncnn::Mat::PIXEL_RGB, rgbImage.cols, rgbImage.rows);

    auto ex = m_net.create_extractor();
    if (m_numThreads > 0) {
        ex.set_num_threads(m_numThreads);
    }

    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);

    for (int i = 0; i < out.w; ++i) {
        output[i] = out[i];
    }
}

