#pragma once

#include <memory>

#include "inferenceEngine.h"

class InferenceManager {
public:
    InferenceManager(const EvalInferenceEngine::InferenceEngineType& inferenceType, const std::vector<std::string>& configFiles);

    bool runInference(const std::string& imageDirectory, double& avgTime);
private:
    std::unique_ptr<EvalInferenceEngine> m_inferencePtr = nullptr;
    static std::vector<std::string> getFilesInDir(const std::string &path);
    static std::string exec(const char* cmd);
};