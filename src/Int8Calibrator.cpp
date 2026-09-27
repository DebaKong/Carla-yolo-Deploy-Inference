#include "Int8Calibrator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

// 构造函数
Int8Calibrator::Int8Calibrator(
    const std::string& calibration_dir,
    int batch_size,
    int input_width,
    int input_height,
    const std::string& calibration_cache
)
    : batch_size_(batch_size),
      input_width_(input_width),
      input_height_(input_height),
      current_index_(0),
      input_size_(
        static_cast<std::size_t>(
            batch_size * 3 * input_width * input_height
        )
      ),
      device_input_(nullptr),
      calibration_cache_(calibration_cache)
{
    // 读取Calibration图片
    loadImagePaths(calibration_dir);

    if (image_paths_.empty())
    {
        throw std::runtime_error(
            "No calibration images found in: "
            + calibration_dir
        );
    }

    // Host buffer
    host_input_.resize(input_size_);

    // GPU buffer
    if (
        !checkCuda(
            cudaMalloc(
                &device_input_,
                input_size_ * sizeof(float)
            ),
            "dudaMalloc calibration buffer"
        )
    )
    {
        throw std::runtime_error(
            "Failed to allocate calibration GPU buffer"
        );
    }

    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << "INT8 Calibrator initialized"
        << std::endl;

    std::cout
        << "Calibration images : "
        << image_paths_.size()
        << std::endl;

    std::cout
        << "Batch size         : "
        << batch_size_
        << std::endl;

    std::cout
        << "Input size         : "
        << input_width_
        << " x "
        << input_height_
        << std::endl;

    std::cout
        << "Number of batches  : "
        << image_paths_.size() / batch_size_
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;
}


// 析构函数
Int8Calibrator::~Int8Calibrator() noexcept
{
    if (device_input_ != nullptr)
    {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
}


// 获取BatchSize
int Int8Calibrator::getBatchSize() const noexcept
{
    return batch_size_;
}


// 加载calibration images
void Int8Calibrator::loadImagePaths(
    const std::string& calibration_dir
)
{
    for (const auto& entry : std::filesystem::directory_iterator(calibration_dir))
    {
        if (!entry.is_regular_file()) continue;

        std::string extension = entry.path().extension().string();

        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            ::tolower
        );

        if (extension == ".jpg" ||
            extension == ".jpeg" ||
            extension == ".png" ||
            extension == ".bmp")
        {
            image_paths_.push_back(
                entry.path().string()
            );
        }
    }

    // 保证顺序稳定
    std::sort(image_paths_.begin(), image_paths_.end());
}


// Preprocess，必须与YOLOv8::preprocess()完全一致
void Int8Calibrator::preprocess(
    const cv::Mat& image,
    float* input
)
{
    if (image.empty())
    {
        throw std::runtime_error(
            "Calibration image is empty"
        );
    }
    
    // 1.计算scale
    float scale_x = static_cast<float>(input_width_) / image.cols;
    float scale_y = static_cast<float>(input_height_) / image.rows;
    float scale = std::min(scale_x, scale_y);

    // 2.resize
    int new_width = static_cast<int>(std::round(image.cols * scale));
    int new_height = static_cast<int>(std::round(image.rows * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    // 3.letterbox
    int pad_x = (input_width_ - new_width) / 2;
    int pad_y = (input_height_ - new_height) / 2;

    cv::Mat canvas(
        input_height_,
        input_width_,
        CV_8UC3,
        cv::Scalar(114,114,114)
    );
    resized.copyTo(
        canvas(cv::Rect(pad_x, pad_y, new_width, new_height))
    );

    // 4.BGR -> RGB
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // 5.HWC -> CHW
    // 6.uint8 -> float
    // 7. /255
    const int channel_size = input_height_ * input_width_;

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < input_height_; ++h)
        {
            for (int w = 0; w < input_width_; ++w)
            {
                input[c * channel_size + h * input_height_ + w] = 
                    canvas.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}


// getBatch
bool Int8Calibrator::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings
) noexcept
{
    try
    {
        // TensorRT要求完整batch
        if (current_index_ + batch_size_ > image_paths_.size())
        {
            return false;
        }

        // 读取一个batch
        for (int b = 0; b < batch_size_; ++b)
        {
            const std::string& image_path = image_paths_[
                current_index_ + b
            ];

            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

            if (image.empty())
            {
                std::cerr
                    << "[INT8] Failed to load image: "
                    << image_path
                    << std::endl;

                return false;
            }

            preprocess(
                image,
                host_input_.data()
                + static_cast<std::size_t>(b) * 3 * input_height_ * input_width_
            );
        }

        // CPU -> GPU
        cudaError_t status = cudaMemcpy(
            device_input_,
            host_input_.data(),
            input_size_ * sizeof(float),
            cudaMemcpyHostToDevice
        );

        if (status != cudaSuccess)
        {
            std::cerr
                << "[INT8] cudaMemcpy failed: "
                << cudaGetErrorString(status)
                << std::endl;

            return false;
        }

        // 返回GPU地址
        bindings[0] = device_input_;
        current_index_ += batch_size_;

        std::cout
            << "\r[INT8 Calibration] "
            << current_index_
            << " / "
            << image_paths_.size()
            << std::flush;

        if (current_index_ ==
            image_paths_.size())
        {
            std::cout
                << std::endl;
        }

        return true;
    }
    catch(...)
    {
        return false;
    }
}


// readCalibrationCache
const void* Int8Calibrator::readCalibrationCache(
    std::size_t& length
) noexcept
{
    length = 0;
    calibration_cache_data_.clear();
    
    std::ifstream file(
        calibration_cache_,
        std::ios::binary
    );

    if (!file.is_open()) return nullptr;

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();

    if (size <= 0) return nullptr;

    file.seekg(0, std::ios::beg);
    calibration_cache_data_.resize(static_cast<std::size_t>(size));
    file.read(calibration_cache_data_.data(), size);

    if (!file)
    {
        calibration_cache_data_.clear();
        return nullptr;
    }

    length = calibration_cache_data_.size();

    std::cout
        << "[INT8] Calibration cache loaded: "
        << calibration_cache_
        << std::endl;

    return calibration_cache_data_.data();
}


// writeCalibrationCache
void Int8Calibrator::writeCalibrationCache(
    const void* cache,
    std::size_t length
) noexcept
{
    if (cache == nullptr || length == 0) return;

    std::ofstream file(
        calibration_cache_,
        std::ios::binary
    );

    if (!file.is_open())
    {
        std::cerr
            << "[INT8] Failed to write calibration cache: "
            << calibration_cache_
            << std::endl;

        return;
    }

    file.write(
        static_cast<const char*>(cache),
        length
    );

    file.close();
    std::cout
        << "[INT8] Calibration cache saved: "
        << calibration_cache_
        << std::endl;
}


//CUDA check
bool Int8Calibrator::checkCuda(
    cudaError_t status,
    const char* message
) const noexcept
{
    if (status != cudaSuccess)
    {
        std::cerr
            << "[CUDA ERROR] "
            << message
            << " : "
            << cudaGetErrorString(status)
            << std::endl;

        return false;
    }

    return true;
}
