cmake_minimum_required(VERSION 3.17)


project(deeplearningInferenceBenchmarks)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -DNDEBUG -fopenmp")

find_package(OpenCV REQUIRED)

include_directories(src/util)
include_directories(src/inference)
include_directories(src)
include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_amd64/install/include/ncnn")

link_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_amd64/install/lib")



set(SOURCE_FILES
        src/main.cpp
        src/util/util.cpp
        src/inference/inferenceEngineTemplate.h
        src/inference/ncnnInferEng.cpp
        src/inferenceManager.cpp
        )

add_executable(deeplearningInferenceBenchmarks ${SOURCE_FILES})
target_link_libraries(deeplearningInferenceBenchmarks ${OpenCV_LIBS} ncnn)