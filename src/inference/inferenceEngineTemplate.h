#pragma once

#include <array>
#include <opencv2/opencv.hpp>

class InferenceEngineTemplate {
public:
    InferenceEngineTemplate() = default;
    virtual ~InferenceEngineTemplate() = default;

    // Run inference and copy the output
    virtual void runInference(const cv::Mat& rgbImage, std::array<float, 500>& output) = 0;
};