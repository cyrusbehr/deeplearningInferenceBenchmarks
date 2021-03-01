cmake_minimum_required(VERSION 3.17)

option(USE_MXNET "Use the mxnet backend" OFF)
option(USE_ONNX_DEFAULT "Use the onnx-runtime defualt CPU backend" ON)

set(USE_ONNX_DEFAULT ON) # TODO Remove this
add_definitions(-DUSE_ONNX_DEFAULT) # TODO remove this

project(deeplearningInferenceBenchmarks)

if (USE_MXNET)
    add_definitions(-DUSE_MXNET)
elseif(USE_ONNX_DEFAULT)
    add_definitions(-DUSE_ONNX_DEFAULT)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -DNDEBUG -fopenmp")

find_package(OpenCV REQUIRED)

include_directories(src/util)
include_directories(src/inference)
include_directories(src)
include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_amd64/install/include/ncnn")

link_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_amd64/install/lib")

if (USE_MXNET)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/incubator-mxnet/build_amd64/packaged/usr/local/include)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/incubator-mxnet/build_amd64)
elseif(USE_ONNX_DEFAULT)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/onnxruntime-linux-x64-1.6.0/include)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/onnxruntime-linux-x64-1.6.0/lib)
endif()

set(LINKER_LIBS ${OpenCV_LIBS})

set(SOURCE_FILES
        src/main.cpp
        src/util/util.cpp
        src/inference/inferenceEngineTemplate.h
        src/inferenceManager.cpp
        )

if (USE_MXNET)
    list(APPEND SOURCE_FILES src/inference/mxnetInferEng.cpp)
    list(APPEND LINKER_LIBS libmxnet.so)
elseif(USE_ONNX_DEFAULT)
    list(APPEND SOURCE_FILES src/inference/onnxRuntimeDefaultInferEng.cpp)
    list(APPEND LINKER_LIBS libonnxruntime.so)
else()
    list(APPEND SOURCE_FILES src/inference/ncnnInferEng.cpp)
    list(APPEND LINKER_LIBS ncnn)
endif()


add_executable(deeplearningInferenceBenchmarks ${SOURCE_FILES})
target_link_libraries(deeplearningInferenceBenchmarks ${LINKER_LIBS})