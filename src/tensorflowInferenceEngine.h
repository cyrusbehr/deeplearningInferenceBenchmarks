#pragma once

#ifdef USE_TENSORFLOW
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;
#endif

#include "inferenceEngine.h"

#ifdef USE_TENSORFLOW
class TensorFlowInferenceEngine
        : public EvalInferenceEngine {
public:
    TensorFlowInferenceEngine(const std::string& modelPath);

    ~TensorFlowInferenceEngine() override;
    bool runInference(const cv::Mat& imgRGB) override;
private:
    tensorflow::Status loadModel(tensorflow::Session *sess, std::string graph_fn, std::string checkpoint_fn = "");
    tensorflow::Session* m_sess;
    char m_ompNumThreads[18];

};
#endif
