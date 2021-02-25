#pragma once

#include <memory>

#include "ncnnInferEng.h" // TODO will need to put ifdef statements here

class InferenceManager {
public:
    InferenceManager(const std::string& modelDir);

    void runBenchmark(unsigned int numIterations = 1000);
private:
    std::unique_ptr<InferenceEngine> m_inferenceEnginePtr = nullptr;
    const std::string m_imagePath;

    void readTemplateFromDisk(const std::string& templatePath, std::array<float, 500>& templ);
    float dotProduct(const std::array<float, 500>& v1, const std::array<float, 500>& v2);
    void normalize(std::array<float, 500>& v);
};