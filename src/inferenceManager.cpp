#include <chrono>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sstream>

#include "inferenceManager.h"
#include "onnxruntimeInferenceEngine.h"
#include "mxnetInferenceEngine.h"
#include "ncnnInferenceEngine.h"
#include "tvmInferenceEngine.h"
#include "openVinoInferenceEngine.h"
#include "tensorflowInferenceEngine.h"

typedef std::chrono::high_resolution_clock Clock;
namespace fs = std::experimental::filesystem;

InferenceManager::InferenceManager(const EvalInferenceEngine::InferenceEngineType& inferenceType, const std::vector<std::string>& configFiles) {
    if (inferenceType == EvalInferenceEngine::InferenceEngineType::NCNN) {
        m_inferencePtr = std::make_unique<NcnnInferenceEngine>(configFiles[0], configFiles[1]);
#ifdef USE_MXNET
    } else if (inferenceType == EvalInferenceEngine::InferenceEngineType::MXNET) {
            m_inferencePtr = std::make_unique<MxnetInferenceEngine>(configFiles[0], configFiles[1]);
#endif
#ifdef USE_TVM
    } else if (inferenceType == EvalInferenceEngine::InferenceEngineType::TVM) {
        m_inferencePtr = std::make_unique<TvmInferenceEngine>(configFiles[0], configFiles[1], configFiles[2]);
#endif
#ifdef USE_ONNX
    } else if (inferenceType == EvalInferenceEngine::InferenceEngineType::ONNXRUNTIME) {
        m_inferencePtr = std::make_unique<OnnxruntimeInferenceEngine>(configFiles[0]);
#endif
#ifdef USE_OPENVINO
    } else if (inferenceType == EvalInferenceEngine::InferenceEngineType::OPENVINO) {
        m_inferencePtr = std::make_unique<OpenVinoInferenceEngine>(configFiles[0], configFiles[1]);
#endif
#ifdef USE_TENSORFLOW
    } else if (inferenceType == EvalInferenceEngine::InferenceEngineType::TENSORFLOW) {
        m_inferencePtr = std::make_unique<TensorFlowInferenceEngine>(configFiles[0]);
#endif
    } else {
        throw std::runtime_error("Unsupported inference engine type");
    }
}

bool InferenceManager::runInference(const std::string& imageDirectory, double& avgTime) {
    auto imageList = getFilesInDir(imageDirectory);

    double totalTime = 0.0;
    int numImagesProcessed = 0;

    for (size_t i = 0; i < imageList.size(); ++i) {
        std::cout << "Processing image " << i << "/" << imageList.size() << '\n';
        const auto& imagePath = imageList[i];

        auto img = cv::imread(imagePath);
        // Convert the image to rgb
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        auto t1 = Clock::now();
        auto res = m_inferencePtr->runInference(img);
        auto t2 = Clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        if (!res) {
            continue;
        }

        if (i > 1) {
            // Discard the first result
            totalTime += time;
            ++numImagesProcessed;
        }
    }

    // Ensure we are only using 1 thread
    pid_t pid = getpid();
    std::string command = "cat /proc/" + std::to_string(pid) + "/status | grep Threads";
    auto retStr = exec(command.c_str());

    auto numThreads = retStr.substr(9);
    std::stringstream ss(numThreads);

    int nThreads;
    ss >> nThreads;

    std::cout << "Threads: " << nThreads << std::endl;

    avgTime = totalTime / numImagesProcessed;

    return true;
}

std::vector<std::string> InferenceManager::getFilesInDir(const std::string& path) {
    fs::recursive_directory_iterator iter(path);
    fs::recursive_directory_iterator end;
    std::vector<std::string> listOfFiles;

    while(iter != end) {
        if (!fs::is_directory(iter->path())) {
            listOfFiles.push_back(iter->path().string());
        }
        std::error_code ec;
        iter.increment(ec);
        if (ec) {
            std::string errMsg = "Error While Accessing : " + iter->path().string() + " :: " + ec.message() + '\n';
            throw std::invalid_argument(errMsg);
        }
    }
    return listOfFiles;
}

std::string InferenceManager::exec(const char* cmd) {
    std::array<char, 128> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}
