cmake_minimum_required(VERSION 3.18)

project(self_driving_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

set(BYTETRACK_ROOT
    ${CMAKE_SOURCE_DIR}/third_party/ByteTrack/deploy/TensorRT/cpp
)

include_directories(
    include
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${BYTETRACK_ROOT}/include
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)

# TCP Server
add_executable(
    tcp_server

    src/main.cpp
    src/tcp_server.cpp
    src/protocol.cpp
    src/yolov8.cpp
    src/tracker.cpp

    ${BYTETRACK_ROOT}/src/BYTETracker.cpp
    ${BYTETRACK_ROOT}/src/STrack.cpp
    ${BYTETRACK_ROOT}/src/kalmanFilter.cpp
    ${BYTETRACK_ROOT}/src/lapjv.cpp
    ${BYTETRACK_ROOT}/src/utils.cpp
)

target_link_libraries(
    tcp_server

    nvinfer
    nvinfer_plugin
    cudart

    ${OpenCV_LIBS}
)

# INT8 Engine Builder
add_executable(
    build_int8_engine

    src/build_int8_engine.cpp
    src/Int8Calibrator.cpp
)

target_link_libraries(
    build_int8_engine

    nvinfer
    nvinfer_plugin
    cudart
    nvonnxparser

    ${OpenCV_LIBS}
)

# PPQ INT8 Engine Builder
add_executable(
    ppq_build_int8_engine

    src/PPQInt8EngineBuilder.cpp
)

target_link_libraries(
    ppq_build_int8_engine

    nvinfer
    nvinfer_plugin
    cudart
    nvonnxparser
)

# Evaluate
add_executable(
    evaluate_map
    src/evaluate_map.cpp
    src/yolov8.cpp
)

target_link_libraries(
    evaluate_map
    nvinfer
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
)