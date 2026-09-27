#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_h

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class Int8Calibrator
    : public nvinfer1::IInt8EntropyCalibrator2
{
public:

    Int8Calibrator(
        const std::string& calibration_dir,
        int batch_size,
        int input_width,
        int input_height,
        const std::string& calibration_cache
    );

    ~Int8Calibrator() noexcept override;

    int getBatchSize() const noexcept override;

    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    const void* readCalibrationCache(
        std::size_t& length
    ) noexcept override;

    void writeCalibrationCache(
	const void* cache,
	std::size_t length
    ) noexcept override;

private:

    std::vector<std::string> image_paths_;

    int batch_size_;
    int input_width_;
    int input_height_;

    std::size_t current_index_;
    std::size_t input_size_;

    void* device_input_;

    std::vector<float> host_input_;

    std::string calibration_cache_;
    std::vector<char> calibration_cache_data_;

private:

    void loadImagePaths(
        const std::string& calibration_dir
    );

    void preprocess(
        const cv::Mat& image,
        float* input
    );

    bool checkCuda(
        cudaError_t status,
        const char* message
    ) const noexcept;
};

#endif
