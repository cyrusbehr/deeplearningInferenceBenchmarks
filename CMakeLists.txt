cmake_minimum_required(VERSION 3.17)

option(USE_MXNET "Use the mxnet backend" ON)

set(USE_MXNET true)
project(deeplearningInferenceBenchmarks)

add_definitions(-DUSE_MXNET) # TODO Remove this

if (USE_MXNET)
    add_definitions(-DUSE_MXNET)
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
else()
    list(APPEND SOURCE_FILES src/inference/ncnnInferEng.cpp)
    list(APPEND LINKER_LIBS ncnn)
endif()


add_executable(deeplearningInferenceBenchmarks ${SOURCE_FILES})
target_link_libraries(deeplearningInferenceBenchmarks ${LINKER_LIBS})