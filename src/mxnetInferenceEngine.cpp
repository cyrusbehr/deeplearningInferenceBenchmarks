#ifdef USE_MXNET
#include "mxnetInferenceEngine.h"

MxnetInferenceEngine::MxnetInferenceEngine(const std::string& binFilepath, const std::string& paramFilepath)
        :m_globalCtx(Context::cpu())
{
    m_net = Symbol::Load(binFilepath).GetInternals()["fc1_output"];
    std::map<std::string, NDArray> parameters;
    NDArray::Load(paramFilepath, nullptr, &parameters);

    for (const auto &k : parameters) {
        if (k.first.substr(0, 4) == "aux:") {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_auxMap[name] = k.second.Copy(m_globalCtx);
        }
        if (k.first.substr(0, 4) == "arg:") {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_argsMap[name] = k.second.Copy(m_globalCtx);
        }
    }

    m_argsMap["data"] = NDArray(Shape(1, 3, 112, 112), m_globalCtx, false);
    m_executor = m_net.SimpleBind(m_globalCtx, m_argsMap, std::map<std::string, NDArray>(),
                                  std::map<std::string, OpReqType>(), m_auxMap);
}

MxnetInferenceEngine::~MxnetInferenceEngine() {
    delete m_executor;
}

bool MxnetInferenceEngine::runInference(const cv::Mat& imgRGB) {
    const auto faceArr = mtxToNDArr(imgRGB);
    faceArr.CopyTo(&(m_executor->arg_dict()["data"]));
    m_executor->Forward(false);
    const auto outputArr = m_executor->outputs[0].Copy(Context(kCPU, 0));

    return outputArr.Size() != 0;
}

NDArray MxnetInferenceEngine::mtxToNDArr(const cv::Mat& frame) {
    NDArray retArr(Shape(1, frame.channels(), frame.rows, frame.cols), m_globalCtx, false);
    std::vector<float> data;

    for (int c = 0; c < frame.channels(); ++c) {
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                data.emplace_back(static_cast<float>(frame.data[(i * frame.rows + j) * 3 + c]));
            }
        }
    }

    retArr.SyncCopyFromCPU(data.data(), 1 * frame.channels() * frame.rows * frame.cols);
    // NDArray::WaitAll();

    return retArr;
}
#endif