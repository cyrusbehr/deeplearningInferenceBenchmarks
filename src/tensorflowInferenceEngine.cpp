#include "tensorflowInferenceEngine.h"

#ifdef USE_TENSORFLOW
TensorFlowInferenceEngine::TensorFlowInferenceEngine(const std::string& modelPath)
        : m_ompNumThreads("OMP_NUM_THREADS=1") {
    putenv(m_ompNumThreads);
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto & config = options.config;
    config.set_inter_op_parallelism_threads(1);
    config.set_intra_op_parallelism_threads(1);
    config.set_use_per_session_threads(false);

    TF_CHECK_OK(tensorflow::NewSession(options, &m_sess));

    TF_CHECK_OK(loadModel(m_sess, modelPath));
}

TensorFlowInferenceEngine::~TensorFlowInferenceEngine() = default;

bool TensorFlowInferenceEngine::runInference(const cv::Mat& imgRGB) {
    // TODO the output from this is not correct
    // May need to convert the input cv mat to float layout

    tensorflow::TensorShape dataShape({1, 112, 112, 3});
    tensorflow::Tensor data(tensorflow::DT_FLOAT, dataShape);

    auto data_ = data.flat<float>().data();
    for (size_t i = 0; i < imgRGB.total(); i+=4) {
        data_[i] = *(reinterpret_cast<const float*>(imgRGB.data + i));
    }

    tensor_dict feed_dict = {
            {"data", data},
    };

    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(m_sess->Run(feed_dict, {"fc1/add_1"},
                          {}, &outputs));

    return true;
}


tensorflow::Status TensorFlowInferenceEngine::loadModel(tensorflow::Session *sess, std::string graph_fn, std::string checkpoint_fn) {
    tensorflow::Status status;

    // Read in the protobuf graph we exported
    tensorflow::MetaGraphDef graph_def;


    status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
    if (status != tensorflow::Status::OK()) return status;

    // create the graph in the current session
    status = sess->Create(graph_def.graph_def());
    if (status != tensorflow::Status::OK()) return status;

    // restore model from checkpoint, iff checkpoint is given
    if (checkpoint_fn != "") {
        const std::string restore_op_name = graph_def.saver_def().restore_op_name();
        const std::string filename_tensor_name =
                graph_def.saver_def().filename_tensor_name();

        tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
                                           tensorflow::TensorShape());
        filename_tensor.scalar<std::string>()() = checkpoint_fn;

        tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};
        status = sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
        if (status != tensorflow::Status::OK()) return status;
    } else {
        status = sess->Run({}, {}, {"init"}, nullptr);
        if (status != tensorflow::Status::OK()) return status;
    }

    return tensorflow::Status::OK();
}
#endif