#include <iostream>
#include "cxxopts.hpp"
#include "inferenceManager.h"

int main(int argc, char* argv[]) {
    const std::string imageDir = "../images";

    cxxopts::Options options(argv[0], "Run inference speed benchmarks");
    options.add_options()
            ("n,ncnn", "Run inference using Ncnn")
            ("m,mxnet", "Run inference using Mxnet")
            ("t,tvm", "Run inference using tvm")
            ("f,tensorflow", "Run inference using tensorflow")
            ("o,onnxruntime", "Run inference using onnxruntime")
            ("v,openvino", "Run inference using openVino")
            ("help", "Print help");

            auto results = options.parse(argc, argv);

    if (results.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (results.count("n")) {
        if (results["n"].as<bool>()) {
            std::cout << "Running inference with ncnn\n";

            const std::string binFilepath = "../models/ncnn.bin";
            const std::string paramFilepath = "../models/ncnn.param";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::NCNN, std::vector<std::string>{binFilepath, paramFilepath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with Ncnn: " << avgTime << '\n';

            return 0;
        }
    }

    if (results.count("m")) {
        if (results["m"].as<bool>()) {
#ifndef USE_MXNET
            std::cout << "Executable has not been built with mxnet support\n";
            std::cout << "Rebuilt with -DUSE_MXNET=ON" << std::endl;
            return -1;
#endif
            std::cout << "Running inference with mxnet\n";

            const std::string binFilepath = "../models/mxnet.bin";
            const std::string paramFilepath = "../models/mxnet.param";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::MXNET, std::vector<std::string>{binFilepath, paramFilepath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with Mxnet: " << avgTime << '\n';

            return 0;
        }
    }

    if (results.count("t")) {
        if (results["t"].as<bool>()) {
#ifndef USE_TVM
            std::cout << "Executable has not been built with tvm support\n";
            std::cout << "Rebuilt with -DUSE_TVM=ON" << std::endl;
            return -1;
#endif
            std::cout << "Running inference with TVM\n";

            const std::string paramsFilepath = "../models/tvm.params";
            const std::string jsonFilepath = "../models/tvm.json";
            const std::string libFilepath = "../models/tvm_lib.so";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::TVM, std::vector<std::string>{paramsFilepath, jsonFilepath, libFilepath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with TVM: " << avgTime << '\n';

            return 0;
        }
    }

    if (results.count("o")) {
        if (results["o"].as<bool>()) {
#ifndef USE_ONNX
            std::cout << "Executable has not been built with onnx support\n";
            std::cout << "Rebuilt with -DUSE_ONNX=ON" << std::endl;
            return -1;
#endif

            std::cout << "Running inference with onnxruntime\n";

            const std::string modelFilepath = "/home/nchafni/Cyrus/models/faceRecognition/Insightface/model-r100-ii/updated_resnet100.onnx";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::ONNXRUNTIME, std::vector<std::string>{modelFilepath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with onnxruntime: " << avgTime << '\n';

            return 0;
        }
    }

    if (results.count("v")) {
        if (results["v"].as<bool>()) {
#ifndef USE_OPENVINO
            std::cout << "Executable has not been built with openvino support\n";
            std::cout << "Rebuilt with -DUSE_OPENVINO=ON" << std::endl;
            return -1;
#endif
            std::cout << "Running inference with openVino\n";

            const std::string xmlFilepath = "/opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/model-0000.xml";
            const std::string binFilepath = "/opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/model-0000.bin";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::OPENVINO, std::vector<std::string>{xmlFilepath, binFilepath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with openVino: " << avgTime << '\n';

            return 0;
        }
    }

    if (results.count("f")) {
        if (results["f"].as<bool>()) {
#ifndef USE_TENSORFLOW
            std::cout << "Executable has not been built with tensorflow support\n";
            std::cout << "Rebuilt with -DUSE_TENSORFLOW=ON" << std::endl;
            return -1;
#endif
            std::cout << "Running inference with Tensorflow\n";

            const std::string modelPath = "../models/my-model.meta";

            InferenceManager inferenceManager(EvalInferenceEngine::InferenceEngineType::TENSORFLOW, std::vector<std::string>{modelPath});
            double avgTime;
            if (!inferenceManager.runInference(imageDir, avgTime)) {
                std::cout << "Unable to run inference\n";
                return -1;
            }

            std::cout << "Inference successful\n";
            std::cout << "Average time with Tensorflow: " << avgTime << '\n';

            return 0;
        }
    }

    std::cout << options.help() << std::endl;
    return -1;

}