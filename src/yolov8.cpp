#include "yolov8.h"

#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

using namespace nvinfer1;


// ============================================================
// TensorRT Logger
// ============================================================

void Logger::log(
    Severity severity,
    const char* msg
) noexcept
{
    // 只输出 WARNING 以及 ERROR
    if (severity <= Severity::kWARNING)
    {
        std::cout
            << "[TensorRT] "
            << msg
            << std::endl;
    }
}


// ============================================================
// CUDA错误检查
// ============================================================

static void checkCuda(
    cudaError_t status,
    const char* message
)
{
    if (status != cudaSuccess)
    {
        std::cerr
            << "[CUDA ERROR] "
            << message
            << " : "
            << cudaGetErrorString(status)
            << std::endl;

        throw std::runtime_error(
            "CUDA operation failed"
        );
    }
}


// ============================================================
// 构造函数
// ============================================================

YOLOv8::YOLOv8(
    const std::string& engine_path
)
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      input_buffer_(nullptr),
      output_buffer_(nullptr),
      stream_(nullptr),
      input_width_(0),
      input_height_(0),
      input_channels_(0),
      input_size_(0),
      output_size_(0),
      scale_(1.0f),
      pad_x_(0),
      pad_y_(0)
{
    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << "Loading TensorRT engine..."
        << std::endl;

    // 1. 加载 Engine
    loadEngine(engine_path);

    // 2. 获取 Tensor 信息
    getTensorInfo();

    // 3. 创建 Execution Context
    context_ =
        engine_->createExecutionContext();

    if (context_ == nullptr)
    {
        throw std::runtime_error(
            "Failed to create TensorRT execution context"
        );
    }

    std::cout
        << "Execution context created successfully."
        << std::endl;

    // 4. 创建 CUDA Stream
    checkCuda(
        cudaStreamCreate(&stream_),
        "cudaStreamCreate"
    );

    std::cout
        << "CUDA stream created successfully."
        << std::endl;

    // 5. 分配 GPU Buffer
    allocateBuffers();

    std::cout
        << "GPU buffers allocated successfully."
        << std::endl;

    std::cout
        << "TensorRT engine initialized successfully."
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;

}


// ============================================================
// 析构函数
// ============================================================

YOLOv8::~YOLOv8()
{
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }


    if (input_buffer_)
    {
        cudaFree(input_buffer_);
        input_buffer_ = nullptr;
    }


    if (output_buffer_)
    {
        cudaFree(output_buffer_);
        output_buffer_ = nullptr;
    }


    // TensorRT 10.x
    // 不再使用 destroy()
    delete context_;
    context_ = nullptr;

    delete engine_;
    engine_ = nullptr;

    delete runtime_;
    runtime_ = nullptr;
}


// ============================================================
// 加载 Engine
// ============================================================

void YOLOv8::loadEngine(
    const std::string& engine_path
)
{
    std::ifstream file(
        engine_path,
        std::ios::binary
    );


    if (!file.is_open())
    {
        throw std::runtime_error(
            "Cannot open TensorRT engine file: "
            + engine_path
        );
    }


    // 移动到文件末尾
    file.seekg(
        0,
        std::ios::end
    );


    std::streampos file_size =
        file.tellg();


    // 防止之前出现的
    // vector larger than max_size()
    if (file_size <= 0)
    {
        throw std::runtime_error(
            "Engine file is empty or invalid: "
            + engine_path
        );
    }


    // 回到文件开头
    file.seekg(
        0,
        std::ios::beg
    );


    size_t size =
        static_cast<size_t>(
            file_size
        );


    std::vector<char> engine_data(
        size
    );


    file.read(
        engine_data.data(),
        size
    );


    if (!file)
    {
        throw std::runtime_error(
            "Failed to read engine file"
        );
    }


    file.close();


    std::cout
        << "Engine file size: "
        << size / 1024.0 / 1024.0
        << " MB"
        << std::endl;


    // ========================================================
    // TensorRT 10.8
    // ========================================================

    runtime_ =
        createInferRuntime(logger);


    if (runtime_ == nullptr)
    {
        throw std::runtime_error(
            "Failed to create TensorRT runtime"
        );
    }


    engine_ =
        runtime_->deserializeCudaEngine(
            engine_data.data(),
            size
        );


    if (engine_ == nullptr)
    {
        throw std::runtime_error(
            "Failed to deserialize TensorRT engine"
        );
    }


    std::cout
        << "Engine deserialized successfully."
        << std::endl;
}


// ============================================================
// 获取 Tensor 信息
// ============================================================

void YOLOv8::getTensorInfo()
{
    int nbIOTensors =
        engine_->getNbIOTensors();


    std::cout
        << "Number of IO tensors: "
        << nbIOTensors
        << std::endl;


    for (int i = 0; i < nbIOTensors; ++i)
    {
        const char* name =
            engine_->getIOTensorName(i);


        TensorIOMode mode =
            engine_->getTensorIOMode(name);


        Dims dims =
            engine_->getTensorShape(name);


        std::cout
            << "Tensor "
            << i
            << ": "
            << name
            << std::endl;


        std::cout
            << "  Shape: ";


        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout
                << dims.d[j];


            if (j != dims.nbDims - 1)
            {
                std::cout << " x ";
            }
        }


        std::cout
            << std::endl;


        std::cout
            << "  Type: ";


        if (engine_->getTensorDataType(name)
            == DataType::kFLOAT)
        {
            std::cout
                << "FLOAT";
        }
        else if (
            engine_->getTensorDataType(name)
            == DataType::kHALF
        )
        {
            std::cout
                << "HALF";
        }
        else
        {
            std::cout
                << "Other";
        }


        std::cout
            << std::endl;


        if (mode == TensorIOMode::kINPUT)
        {
            input_name_ =
                name;

            if (dims.nbDims != 4)
            {
                throw std::runtime_error(
                    "Unexpected input dimensions"
                );
            }


            input_channels_ =
                dims.d[1];

            input_height_ =
                dims.d[2];

            input_width_ =
                dims.d[3];


            input_size_ =
                static_cast<size_t>(
                    input_channels_
                    * input_height_
                    * input_width_
                );
        }
        else if (
            mode == TensorIOMode::kOUTPUT
        )
        {
            output_name_ =
                name;


            if (dims.nbDims != 3)
            {
                throw std::runtime_error(
                    "Unexpected output dimensions"
                );
            }


            output_size_ = 1;


            for (int j = 0;
                 j < dims.nbDims;
                 ++j)
            {
                output_size_ *=
                    static_cast<size_t>(
                        dims.d[j]
                    );
            }
        }
    }


    if (input_name_.empty())
    {
        throw std::runtime_error(
            "Input tensor not found"
        );
    }


    if (output_name_.empty())
    {
        throw std::runtime_error(
            "Output tensor not found"
        );
    }


    std::cout
        << std::endl;


    std::cout
        << "Input tensor : "
        << input_name_
        << std::endl;


    std::cout
        << "Output tensor: "
        << output_name_
        << std::endl;


    std::cout
        << "Input size   : "
        << input_size_
        << " floats"
        << std::endl;


    std::cout
        << "Output size  : "
        << output_size_
        << " floats"
        << std::endl;
}


// ============================================================
// 分配 GPU Buffer
// ============================================================

void YOLOv8::allocateBuffers()
{
    size_t input_bytes =
        input_size_
        * sizeof(float);


    size_t output_bytes =
        output_size_
        * sizeof(float);


    std::cout
        << "Allocating GPU memory..."
        << std::endl;


    checkCuda(
        cudaMalloc(
            &input_buffer_,
            input_bytes
        ),
        "cudaMalloc input"
    );


    checkCuda(
        cudaMalloc(
            &output_buffer_,
            output_bytes
        ),
        "cudaMalloc output"
    );


    std::cout
        << "Input GPU memory : "
        << input_bytes / 1024.0 / 1024.0
        << " MB"
        << std::endl;


    std::cout
        << "Output GPU memory: "
        << output_bytes / 1024.0 / 1024.0
        << " MB"
        << std::endl;
}


// ============================================================
// YOLOv8 前处理
//
// OpenCV:
// BGR HWC uint8
//
//        ↓
//
// Letterbox 640x640
//
//        ↓
//
// RGB
//
//        ↓
//
// float / 255
//
//        ↓
//
// HWC → CHW
// ============================================================

void YOLOv8::preprocess(
    const cv::Mat& image,
    float* input
)
{
    if (image.empty())
    {
        throw std::runtime_error(
            "Input image is empty"
        );
    }


    // --------------------------------------------------------
    // 计算缩放比例
    // --------------------------------------------------------

    float scale_x =
        static_cast<float>(
            input_width_
        )
        / image.cols;


    float scale_y =
        static_cast<float>(
            input_height_
        )
        / image.rows;


    scale_ =
        std::min(
            scale_x,
            scale_y
        );


    int new_width =
        static_cast<int>(
            std::round(
                image.cols * scale_
            )
        );


    int new_height =
        static_cast<int>(
            std::round(
                image.rows * scale_
            )
        );


    // --------------------------------------------------------
    // resize
    // --------------------------------------------------------

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(
            new_width,
            new_height
        )
    );


    // --------------------------------------------------------
    // Letterbox
    // --------------------------------------------------------

    pad_x_ =
        (input_width_ - new_width)
        / 2;


    pad_y_ =
        (input_height_ - new_height)
        / 2;


    cv::Mat canvas(
        input_height_,
        input_width_,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114
        )
    );


    resized.copyTo(
        canvas(
            cv::Rect(
                pad_x_,
                pad_y_,
                new_width,
                new_height
            )
        )
    );


    // --------------------------------------------------------
    // BGR → RGB
    // --------------------------------------------------------

    cv::cvtColor(
        canvas,
        canvas,
        cv::COLOR_BGR2RGB
    );


    // --------------------------------------------------------
    // uint8 → float
    //
    // 同时 HWC → CHW
    // --------------------------------------------------------

    for (int c = 0;
         c < input_channels_;
         ++c)
    {
        for (int h = 0;
             h < input_height_;
             ++h)
        {
            for (int w = 0;
                 w < input_width_;
                 ++w)
            {
                input[
                    c
                    * input_height_
                    * input_width_
                    + h
                    * input_width_
                    + w
                ] =
                    canvas.at<cv::Vec3b>(
                        h,
                        w
                    )[c]
                    / 255.0f;
            }
        }
    }
}


// ============================================================
// 推理
// ============================================================

std::vector<Detection> YOLOv8::infer(
    const cv::Mat& image
)
{
    // ========================================================
    // 1. 创建 CPU 输入
    // ========================================================

    std::vector<float> input(
        input_size_
    );

    // 计时
    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================
    // 2. 前处理
    // ========================================================

    auto preprocess_start = std::chrono::high_resolution_clock::now();

    preprocess(
        image,
        input.data()
    );

    auto preprocess_end = std::chrono::high_resolution_clock::now();

    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

    // ========================================================
    // 3. CPU → GPU
    // ========================================================
    auto h2d_start = std::chrono::high_resolution_clock::now();

    checkCuda(
        cudaMemcpyAsync(
            input_buffer_,
            input.data(),
            input_size_
            * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_
        ),
        "cudaMemcpyAsync input"
    );

    checkCuda(
	cudaStreamSynchronize(stream_),
	"cudaStreamSynchronize H2D"
    );

    auto h2d_end = std::chrono::high_resolution_clock::now();

    double h2d_time = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();

    // ========================================================
    // 4. TensorRT 10.x
    //
    // 不使用 binding index
    // 不使用 enqueueV2
    //
    // 使用：
    // setTensorAddress()
    // enqueueV3()
    // ========================================================

    bool input_address_set =
        context_->setTensorAddress(
            input_name_.c_str(),
            input_buffer_
        );


    bool output_address_set =
        context_->setTensorAddress(
            output_name_.c_str(),
            output_buffer_
        );


    if (!input_address_set ||
        !output_address_set)
    {
        throw std::runtime_error(
            "Failed to set TensorRT tensor address"
        );
    }


    // ========================================================
    // 5. TensorRT 推理
    // ========================================================
    auto inference_start =
        std::chrono::high_resolution_clock::now();


    bool success =
        context_->enqueueV3(
            stream_
        );


    if (!success)
    {
        throw std::runtime_error(
            "TensorRT enqueueV3 failed"
        );
    }


    checkCuda(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize inference"
    );


    auto inference_end =
        std::chrono::high_resolution_clock::now();


    double inference_time =
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

    // ========================================================
    // 6. GPU → CPU
    // ========================================================
    auto d2h_start =
        std::chrono::high_resolution_clock::now();

    std::vector<float> output(
        output_size_
    );


    checkCuda(
        cudaMemcpyAsync(
            output.data(),
            output_buffer_,
            output_size_
            * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        ),
        "cudaMemcpyAsync output"
    );

    checkCuda(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize D2H"
    );


    auto d2h_end =
        std::chrono::high_resolution_clock::now();


    double d2h_time =
        std::chrono::duration<double, std::milli>(
            d2h_end - d2h_start
        ).count();

    // ========================================================
    // 7. 等待 CUDA stream
    // ========================================================

    //checkCuda(
    //    cudaStreamSynchronize(
    //        stream_
    //    ),
    //    "cudaStreamSynchronize"
    //);


    // ========================================================
    // 8. 后处理
    // ========================================================
    auto postprocess_start =
        std::chrono::high_resolution_clock::now();


    std::vector<Detection> detections =
        postprocess(
            output,
            image
        );


    auto postprocess_end =
        std::chrono::high_resolution_clock::now();


    double postprocess_time =
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start
        ).count();

    // 所有计时统计
    auto total_end =
        std::chrono::high_resolution_clock::now();


    double total_time =
        std::chrono::duration<double, std::milli>(
            total_end - total_start
        ).count();

    static int profile_count = 0;
    profile_count++;

    if (profile_count <= 5)
    {
        std::cout
	    << std::fixed
            << std::setprecision(3)
            << "[PROFILE] "
            << "Preprocess: "
            << preprocess_time
            << " ms | "
            << "H2D: "
            << h2d_time
            << " ms | "
            << "TensorRT: "
            << inference_time
            << " ms | "
            << "D2H: "
            << d2h_time
            << " ms | "
            << "Postprocess: "
            << postprocess_time
            << " ms | "
            << "Total: "
            << total_time
            << " ms"
            << std::endl;
    }

    return detections;

}

// ============================================================
// 后处理
//
// output[1][10][8400]
//
// ↓
//
// box decode
//
// ↓
//
// confidence filtering
//
// ↓
//
// class selection
//
// ↓
//
// NMS
// ============================================================

std::vector<Detection> YOLOv8::postprocess(
    const std::vector<float>& output,
    const cv::Mat& image
)
{
    std::vector<Detection> detections;

    // 参数
    const float CONF_THRESHOLD = 0.25f;
    const float NMS_THRESHOLD = 0.45f;
    const int num_attributes = 84;
    const int num_classes = num_attributes - 4;
    const int num_boxes = 8400;

    if (output.size() < static_cast<size_t>(num_attributes * num_boxes))
    {
	throw std::runtime_error("Invaild YOLOv8 output size");
    }

    // 临时保存NMS前的结果
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    boxes.reserve(num_boxes);
    confidences.reserve(num_boxes);
    class_ids.reserve(num_boxes);

    // 遍历8400个候选框
    for (int i = 0; i < num_boxes; ++i)
    {
	float cx = output[0*num_boxes + i];
	float cy = output[1*num_boxes + i];
	float w = output[2*num_boxes + i];
	float h = output[3*num_boxes + i];

	//找到最高类别概率
	float max_confidence = 0.0f;
	int max_class_id = -1;

	for (int c = 0; c < num_classes; ++c)
	{
	    float confidence = output[(4+c)*num_boxes + i];

	    if (confidence > max_confidence)
	    {
		max_confidence = confidence;
		max_class_id = c;
	    }
	}

	//置信度过滤
	if (max_confidence < CONF_THRESHOLD)
	{
	    continue;
	}

	if (max_class_id < 0)
	{
	    continue;
	}

	// xywh -> xyxy
	float x1 = cx - w * 0.5f;
	float y1 = cy - h * 0.5f;
	float x2 = cx + w * 0.5f;
	float y2 = cy + h * 0.5f;

	// 去除Letterbox padding
	x1 -= static_cast<float>(pad_x_);
	x2 -= static_cast<float>(pad_x_);
	y1 -= static_cast<float>(pad_y_);
	y2 -= static_cast<float>(pad_y_);

	// 映射回原始图像
	x1 /= scale_;
	x2 /= scale_;
	y1 /= scale_;
	y2 /= scale_;

	// 限制到原图范围
	x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols - 1)));
	x2 = std::max(0.0f, std::min(x2, static_cast<float>(image.cols - 1)));
	y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows - 1)));
	y2 = std::max(0.0f, std::min(y2, static_cast<float>(image.rows - 1)));

	// 转换成OpenCV Rect
	int left = static_cast<int>(std::round(x1));
	int top = static_cast<int>(std::round(y1));
	int width = static_cast<int>(std::round(x2 - x1));
	int height = static_cast<int>(std::round(y2 - y1));

	// 检查框是否合法
	if (width <= 0 || height <= 0)
	{
	    continue;
	}

	// 保存
	boxes.emplace_back(left, top, width, height);
	confidences.push_back(max_confidence);
	class_ids.push_back(max_class_id);
    }

    // 无检测结果，返回空
    if (boxes.empty())
    {
	return detections;
    }

    // NMS
    // OpenCV NMSBoxes 默认会把所有类别放在一起NMS
    // 这里按照类别分别执行NMS
    std::vector<int> final_indices;
    std::vector<std::vector<int>> groups(num_classes);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
	int cid = class_ids[i];
	groups[cid].push_back(static_cast<int>(i));
    }

    for (int c = 0; c < num_classes; ++c)
    {
	auto& idx_list = groups[c];

	if (idx_list.empty()) continue;

	std::vector<cv::Rect> class_boxes;
	std::vector<float> class_confs;
	class_boxes.reserve(idx_list.size());
	class_confs.reserve(idx_list.size());

	for (int idx : idx_list)
	{
	    class_boxes.push_back(boxes[idx]);
	    class_confs.push_back(confidences[idx]);
	}

	std::vector<int> nms_indices;

	cv::dnn::NMSBoxes(class_boxes, class_confs, CONF_THRESHOLD, NMS_THRESHOLD, nms_indices);

	for (int n : nms_indices)
	{
	    final_indices.push_back(idx_list[n]);
	}
    }

    // 生成最终Detection
    for (int index : final_indices)
    {
	Detection detection;

	detection.class_id = class_ids[index];
	detection.confidence = confidences[index];
	detection.box = boxes[index];

	detections.push_back(detection);
    }

    // 按置信度从高到低排序
    std::sort(
	detections.begin(),
	detections.end(),
	[](const Detection& a, const Detection& b)
	{
	    return a.confidence > b.confidence;
	}
    );

    return detections;
}
