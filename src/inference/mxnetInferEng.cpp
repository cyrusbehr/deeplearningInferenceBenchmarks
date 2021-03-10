#include "mxnetInferEng.h"

inline NDArray InferenceEng::matToNDArray(cv::Mat rgb_image, Context ctx) {
    std::vector<float> data_buffer;

    // hwc to chw conversion
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rgb_image.rows; ++i) {
            for (int j = 0; j < rgb_image.cols; ++j) {
                data_buffer.push_back(static_cast<float>(rgb_image.data[(i * rgb_image.cols + j) * 3 + c]));
            }
        }
    }

    // construct NDArray from data buffer
    return NDArray(data_buffer, Shape(1, 3, rgb_image.rows, rgb_image.cols), ctx);
}

InferenceEng::InferenceEng(const std::string &modelDir)
    : m_globalCtx(Context::cpu()) {
    std::cout << "Using mxnet inference engine" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    const std::string jsonFilepath = modelDir + JSON_NAME;
    const std::string paramsFilepath = modelDir + PARAMS_NAME;

    m_net = Symbol::Load(jsonFilepath);

    std::map<std::string, NDArray> params = NDArray::LoadToMap(paramsFilepath);

    for (auto iter : params) {
        std::string type = iter.first.substr(0, 4);
        std::string name = iter.first.substr(4);
        if (type == "arg:")
            m_argsMap[name] = iter.second.Copy(m_globalCtx);
        else if (type == "aux:")
            m_auxMap[name] = iter.second.Copy(m_globalCtx);
        else
            continue;
    }
    NDArray::WaitAll();

    m_argsMap["data"] = NDArray(Shape(1, 3, 112, 112), m_globalCtx, false);
    m_executor = m_net.SimpleBind(
            m_globalCtx, m_argsMap, std::map<std::string, NDArray>(),
            std::map<std::string, OpReqType>(), m_auxMap);
}

InferenceEng::~InferenceEng() noexcept {
        delete m_executor;
}

void InferenceEng::runInference(const cv::Mat &rgbImage, std::array<float, 500> &output) {
    auto data = matToNDArray(rgbImage, m_globalCtx);
    data.CopyTo(&(m_executor->arg_dict()["data"]));
    NDArray::WaitAll();

    m_executor->Forward(false);
    auto embedding = m_executor->outputs[0].Copy(m_globalCtx);
    NDArray::WaitAll();

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = embedding.At(0, i);
    }
}