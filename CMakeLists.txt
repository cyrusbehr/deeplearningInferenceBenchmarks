cmake_minimum_required(VERSION 3.12)
project(inference_benchmarks)

option(USE_TENSORFLOW "Link tensorflow" OFF)
option(USE_TVM "Link tvm" OFF)
option(USE_MXNET "Link mxnet" OFF)
option(USE_OPENVINO "Link OpenVINO" OFF)
option(USE_ONNX "Link OnnxRuntime" OFF)

if (USE_TENSORFLOW)
    add_definitions(-DUSE_TENSORFLOW)
    find_package(TensorflowCC REQUIRED)
endif()

if (USE_MXNET)
    add_definitions(-DUSE_MXNET)
endif()

if (USE_ONNX)
    add_definitions(-DUSE_ONNX)
endif()


if (USE_TVM)
    add_definitions(-DUSE_TVM)
endif()

if(USE_OPENVINO)
    add_definitions(-DUSE_OPENVINO=ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -fPIC -Ofast -ffast-math -fopenmp -lpthread -Wall")

find_package(OpenCV REQUIRED)

include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_broadwell/install/include/ncnn")
include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/cxxopts/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/tensorflow/")
include_directories("${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/onnxruntime/include")
include_directories(
        /opt/intel/openvino/deployment_tools/inference_engine/include
        /opt/intel/openvino/deployment_tools/inference_engine/samples/common)

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")


set(LIBNCNN ${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/ncnn/build_broadwell/install/lib/libncnn.a)

if (USE_ONNX)
    set(LIBONNXRUNTIME ${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/onnxruntime/lib/libonnxruntime.so)
endif()

if(USE_OPENVINO)
    link_directories(/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64)
    set(OPEN_VINO_LIBS libinference_engine.so libcpu_extension_avx2.so)
endif()
if (USE_TENSORFLOW)
    set(LIBTENSORFLOW "TensorflowCC::Shared")
endif()

if (USE_MXNET)
    set(LIBMXNET ${CMAKE_CURRENT_LIST_DIR}/3rd_party_libs/mxnet/build/libmxnet.so)
endif()

SET(SOURCE_FILES src/main.cpp src/inferenceEngine.cpp src/inferenceManager.cpp src/onnxruntimeInferenceEngine.cpp src/mxnetInferenceEngine.cpp
        src/ncnnInferenceEngine.cpp src/tvmInferenceEngine.cpp src/tensorflowInferenceEngine.cpp src/openVinoInferenceEngine.cpp)

if (USE_TVM)
    SET(TVM_SOURCE src/tvm_runtime_pack.cc)
endif()

add_executable(inference_benchmarks ${SOURCE_FILES} ${TVM_SOURCE})

target_compile_options(inference_benchmarks PRIVATE -march=broadwell)
target_link_libraries(inference_benchmarks PRIVATE ${LIBMXNET} ${LIBONNXRUNTIME} ${LIBTENSORFLOW} ${OpenCV_LIBS} ${OPEN_VINO_LIBS} stdc++fs ${LIBNCNN} dl)
