#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "Int8Calibrator.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace nvinfer1;


// TensorRT Logger
class Logger : public ILogger
{
public:

    void log(
        Severity severity,
        const char* msg
    ) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout
                << "[TensorRT] "
                << msg
                << std::endl;
        }
    }
};


// RAII deleter

template <typename T>
struct TRTDeleter
{
    void operator()(T* obj) const
    {
        delete obj;
    }
};


// Read binary file

static std::vector<char> readFile(
    const std::string& path
)
{
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Cannot open file: " + path
        );
    }

    file.seekg(
        0,
        std::ios::end
    );

    std::streamsize size =
        file.tellg();

    file.seekg(
        0,
        std::ios::beg
    );

    if (size <= 0)
    {
        throw std::runtime_error(
            "Invalid file: " + path
        );
    }

    std::vector<char> data(
        static_cast<std::size_t>(size)
    );

    file.read(
        data.data(),
        size
    );

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read file: " + path
        );
    }

    return data;
}


// Save TensorRT Engine

static void saveEngine(
    IHostMemory* serialized_engine,
    const std::string& engine_path
)
{
    if (serialized_engine == nullptr)
    {
        throw std::runtime_error(
            "Serialized engine is null"
        );
    }

    std::ofstream file(
        engine_path,
        std::ios::binary
    );

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Cannot open output engine: "
            + engine_path
        );
    }

    file.write(
        static_cast<const char*>(
            serialized_engine->data()
        ),
        serialized_engine->size()
    );

    file.close();

    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << "INT8 engine saved:"
        << std::endl;

    std::cout
        << engine_path
        << std::endl;

    std::cout
        << "Engine size: "
        << serialized_engine->size()
            / 1024.0
            / 1024.0
        << " MB"
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;
}


// Main

int main()
{
    try
    {
        // 路径

        const std::string ONNX_PATH =
            "assets/yolov8n.onnx";

        const std::string ENGINE_PATH =
            "assets/yolov8n_int8.engine";

        const std::string CALIBRATION_DIR =
            "assets/Calibration/images";

        const std::string CALIBRATION_CACHE =
            "assets/calibration.cache";


        // Calibration 参数

        constexpr int INPUT_WIDTH = 640;
        constexpr int INPUT_HEIGHT = 640;

        constexpr int BATCH_SIZE = 8;


        // Logger

        Logger logger;


        // 创建 Builder

        std::unique_ptr<IBuilder, TRTDeleter<IBuilder>>
            builder(
                createInferBuilder(logger)
            );

        if (!builder)
        {
            throw std::runtime_error(
                "Failed to create TensorRT builder"
            );
        }


        // 创建 Network

        std::unique_ptr<INetworkDefinition,
                        TRTDeleter<INetworkDefinition>>
            network(
                builder->createNetworkV2(0)
            );

        if (!network)
        {
            throw std::runtime_error(
                "Failed to create TensorRT network"
            );
        }


        // 创建ONNX Parser

        std::unique_ptr<
            nvonnxparser::IParser,
            TRTDeleter<nvonnxparser::IParser>
        >
            parser(
                nvonnxparser::createParser(
                    *network,
                    logger
                )
            );

        if (!parser)
        {
            throw std::runtime_error(
                "Failed to create ONNX parser"
            );
        }


        // Parse ONNX

        std::cout
            << "========================================"
            << std::endl;

        std::cout
            << "Parsing ONNX..."
            << std::endl;

        std::cout
            << "ONNX: "
            << ONNX_PATH
            << std::endl;

        std::cout
            << "========================================"
            << std::endl;


        if (!parser->parseFromFile(
                ONNX_PATH.c_str(),
                static_cast<int>(
                    ILogger::Severity::kWARNING
                )
            ))
        {
            std::cerr
                << "Failed to parse ONNX:"
                << std::endl;

            for (int i = 0;
                 i < parser->getNbErrors();
                 ++i)
            {
                std::cerr
                    << parser->getError(i)->desc()
                    << std::endl;
            }

            return -1;
        }


        // 检查 Input

        if (network->getNbInputs() != 1)
        {
            throw std::runtime_error(
                "Expected exactly one input tensor"
            );
        }

        ITensor* input =
            network->getInput(0);

        Dims input_dims =
            input->getDimensions();


        std::cout
            << "Input tensor:"
            << std::endl;

        std::cout
            << "Name: "
            << input->getName()
            << std::endl;

        std::cout
            << "Shape: ";

        for (int i = 0;
             i < input_dims.nbDims;
             ++i)
        {
            std::cout
                << input_dims.d[i];

            if (i != input_dims.nbDims - 1)
            {
                std::cout
                    << " x ";
            }
        }

        std::cout
            << std::endl;


        // 检查 Output

        std::cout
            << "Output tensors:"
            << std::endl;

        for (int i = 0;
             i < network->getNbOutputs();
             ++i)
        {
            ITensor* output =
                network->getOutput(i);

            Dims dims =
                output->getDimensions();

            std::cout
                << "  "
                << output->getName()
                << " : ";

            for (int j = 0;
                 j < dims.nbDims;
                 ++j)
            {
                std::cout
                    << dims.d[j];

                if (j != dims.nbDims - 1)
                {
                    std::cout
                        << " x ";
                }
            }

            std::cout
                << std::endl;
        }


        // 创建 BuilderConfig

        std::unique_ptr<
            IBuilderConfig,
            TRTDeleter<IBuilderConfig>
        >
            config(
                builder->createBuilderConfig()
            );

        if (!config)
        {
            throw std::runtime_error(
                "Failed to create builder config"
            );
        }


        // Workspace

        config->setMemoryPoolLimit(
            MemoryPoolType::kWORKSPACE,
            1ULL << 30
        );


        // 开启 INT8

        config->setFlag(
            BuilderFlag::kINT8
        );


        // INT8 Calibrator

        Int8Calibrator calibrator(
            CALIBRATION_DIR,
            BATCH_SIZE,
            INPUT_WIDTH,
            INPUT_HEIGHT,
            CALIBRATION_CACHE
        );

        config->setInt8Calibrator(
            &calibrator
        );


        // Build

        std::cout
            << "Building INT8 TensorRT engine..."
            << std::endl;


        std::unique_ptr<
            IHostMemory,
            TRTDeleter<IHostMemory>
        >
            serialized_engine(
                builder->buildSerializedNetwork(
                    *network,
                    *config
                )
            );


        if (!serialized_engine)
        {
            throw std::runtime_error(
                "Failed to build INT8 engine"
            );
        }


        // 保存 Engine

        saveEngine(
            serialized_engine.get(),
            ENGINE_PATH
        );


        std::cout
            << std::endl
            << "INT8 engine build completed."
            << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << std::endl
            << "[FATAL ERROR] "
            << e.what()
            << std::endl;

        return -1;
    }
}
