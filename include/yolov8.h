#ifndef YOLOV8_H
#define YOLOV8_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace nvinfer1;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


class Logger : public ILogger
{
public:
    void log(
	Severity severity,
	const char* msg
    )noexcept override;
};


class YOLOv8
{
public:

    YOLOv8(
	const std::string& engine_path
    );

    ~YOLOv8();

    std::vector<Detection> infer(
	const cv::Mat& image
    );

private:

    Logger logger;

    //TensorRT对象
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;

    //Tensor名称
    std::string input_name_;
    std::string output_name_;

    //GPU buffer
    void* input_buffer_;
    void* output_buffer_;

    cudaStream_t stream_;

    //Tensor shape
    int input_width_;
    int input_height_;
    int input_channels_;
    size_t input_size_;
    size_t output_size_;

    //Letterbox参数(后处理使用)
    float scale_;
    int pad_x_;
    int pad_y_;


private:

    //Engine
    void loadEngine(
	const std::string& engine_path
    );

    //Tensor信息
    void getTensorInfo();

    // CUDA Buffer
    void allocateBuffers();

    //前处理
    void preprocess(
	const cv::Mat& image,
	float* input
    );

    //后处理
    std::vector<Detection> postprocess(
	const std::vector<float>& output,
	const cv::Mat& image
    );

};

#endif
