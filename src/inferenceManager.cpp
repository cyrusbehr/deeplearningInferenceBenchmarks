#include <opencv2/opencv.hpp>
#include <utility>
#include <array>
#include <chrono>
#include <fstream>
#include <math.h>

typedef std::chrono::high_resolution_clock Clock;

#include "inferenceManager.h"
#include "util.h"

InferenceManager::InferenceManager(const std::string &modelDir) {
    m_inferenceEnginePtr = std::make_unique<InferenceEngine>(modelDir);
}

void InferenceManager::normalize(std::array<float, 500>& v) {
    auto mag = sqrt(dotProduct(v, v));
    for (auto& elem: v) {
        elem = elem / mag;
    }
}

float InferenceManager::dotProduct(const std::array<float, 500>& v1, const std::array<float, 500>& v2) {
    float dp = 0;
    for (size_t i = 0; i < 500; ++i) {
        dp += v1[i] * v2[i];
    }

    return dp;
}

void InferenceManager::readTemplateFromDisk(const std::string& templatePath, std::array<float, 500>& templ) {
    std::ifstream infile(templatePath, std::ifstream::binary);
    infile.read(reinterpret_cast<char*>(templ.data()), 2000);
}

void InferenceManager::runBenchmark(unsigned int numIterations) {
    // Read the face chip image to rgb
    auto bgrImg = cv::imread("../test_data/face_chip.jpg");

    // Ensure the face chip is of the expected size
    if (bgrImg.cols != 112 || bgrImg.rows != 112 || bgrImg.channels() != 3) {
        throw std::runtime_error("Face chip is not of the expected size!");
    }

    cv::Mat rgbImage;
    cv::cvtColor(bgrImg, rgbImage, cv::COLOR_BGR2RGB);

    std::array<float, 500> output{};

    // Run inference once without timing to initialize weights, etc
    m_inferenceEnginePtr->runInference(rgbImage, output);

    // Time the implementation
    auto t1 = Clock::now();
    for (unsigned int i = 0; i < numIterations; ++i) {
        m_inferenceEnginePtr->runInference(rgbImage, output);
    }
    auto t2 = Clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // Ensure the integrity of the output against the expected result.
    std::array<float, 500> expectedOutput{};
    readTemplateFromDisk("../test_data/template.bin", expectedOutput);

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Expected output" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << expectedOutput[0] << std::endl;

    normalize(output);
    normalize(expectedOutput);

    auto simScore = dotProduct(output, expectedOutput);
    std::cout << "Similarity score: " << simScore << std::endl;
    if (simScore > 1.01 || simScore < 0.99) {
        throw std::runtime_error("Error, similarity score is not 1!");
    }

    auto avgInferenceTime = totalTime / numIterations;
    std::cout << "Average inference speed per face chip: " << avgInferenceTime << " ms" << std::endl;

    // Read the number of threads used
    auto numThreadsUsed = getNumThreads();
    std::cout << "Number of threads used: " << numThreadsUsed << std::endl;

    // Read the memory usage
    auto memUsageKb = getProcessMemUsage();
    std::cout << "Memory usage: " << memUsageKb / 1000000 << " Gb" << std::endl;
}