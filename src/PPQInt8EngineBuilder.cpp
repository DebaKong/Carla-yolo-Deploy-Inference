#include "PPQInt8EngineBuilder.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <cmath>


// ============================================================
// Layer 分类
//
// PPQ 的 act_quant_info 只包含"真正被量化"的张量。
// ONNX 图中还有一些算子只是搬运数据(reshape / split ...),
// 它们的输出张量语义上与输入完全一致,PPQ 常常不记录它们。
//
// 对这类算子,我们可以安全地把 range 从已知张量传播到未知张量。
//
// 注意:
//   这里只使用 TensorRT 10.x LayerType 中真实存在的枚举值。
//   TensorRT 10 已经不再有 kRESHAPE / kSPLIT / kEXPAND 等
//   独立枚举值:
//     - ONNX Reshape / Transpose / Flatten -> kSHUFFLE
//     - ONNX Split / Slice                  -> kSLICE
//   写不存在枚举值会直接编译失败。
// ============================================================

namespace
{

/**
 * @brief 输出与输入数值完全相同的算子(保值算子)。
 *
 * 对这类算子:
 *
 *     output range == input range
 */
bool isValuePreservingLayer(
    nvinfer1::LayerType type)
{
    switch (type)
    {
    case nvinfer1::LayerType::kSHUFFLE:
    case nvinfer1::LayerType::kSLICE:
    case nvinfer1::LayerType::kIDENTITY:
    case nvinfer1::LayerType::kCONSTANT:
    case nvinfer1::LayerType::kSQUEEZE:
    case nvinfer1::LayerType::kUNSQUEEZE:
        return true;

    default:
        return false;
    }
}


/**
 * @brief 输出仍然在原输入取值范围之内的算子。
 *
 * 例如 Sigmoid -> [0, 1]、Softmax -> [0, 1]、MaxPool -> 原值。
 *
 * 对这类算子,直接把输入 range 当作输出 range 是安全上界。
 */
bool isBoundedRangeLayer(
    nvinfer1::LayerType type)
{
    switch (type)
    {
    case nvinfer1::LayerType::kACTIVATION:
    case nvinfer1::LayerType::kSOFTMAX:
    case nvinfer1::LayerType::kPOOLING:
        return true;

    default:
        return false;
    }
}


/**
 * @brief 输出数值范围是"所有输入范围之并集"的算子。
 *
 * 例如 Concat / Add / Max,取 max(input ranges) 是安全上界。
 */
bool isUnionRangeLayer(
    nvinfer1::LayerType type)
{
    switch (type)
    {
    case nvinfer1::LayerType::kELEMENTWISE:
    case nvinfer1::LayerType::kCONCATENATION:
    case nvinfer1::LayerType::kREDUCE:
        return true;

    default:
        return false;
    }
}


/**
 * @brief 根据"张量名"推断它与另一个已知张量是否数值等价。
 *
 * PPQ / ONNX 的命名规则非常固定:
 *
 *     /model.2/Split_output_0
 *     /model.2/Split_output_1
 *
 * 多个 Split 输出是同一份数据的不同视图,数值完全相同。
 * 但 PPQ 有时只记录其中一个(通常是 _1),另一个就缺失了。
 *
 * 为了不依赖 ONNX Parser 把 Split 映射成哪一种 LayerType,
 * 这里直接用命名规则兜底:
 *
 *     <prefix>_output_<index>
 *
 * 只要前缀相同,就认为是同一份数据。
 */
bool isIndexedSibling(
    const std::string& a,
    const std::string& b)
{
    const std::string marker = "_output_";

    const std::size_t markerA =
        a.rfind(marker);

    const std::size_t markerB =
        b.rfind(marker);


    if (markerA == std::string::npos ||
        markerB == std::string::npos)
    {
        return false;
    }


    // --------------------------------------------------------
    // 比较 "_output_" 之前的前缀
    // --------------------------------------------------------

    if (a.compare(0, markerA, b, 0, markerB) != 0)
    {
        return false;
    }


    // --------------------------------------------------------
    // 后缀必须都是纯数字
    // --------------------------------------------------------

    const std::string suffixA =
        a.substr(markerA + marker.size());

    const std::string suffixB =
        b.substr(markerB + marker.size());


    if (suffixA.empty() ||
        suffixB.empty())
    {
        return false;
    }


    for (char c : suffixA)
    {
        if (c < '0' || c > '9')
        {
            return false;
        }
    }


    for (char c : suffixB)
    {
        if (c < '0' || c > '9')
        {
            return false;
        }
    }


    // --------------------------------------------------------
    // 索引必须不同,否则就是同一个张量
    // --------------------------------------------------------

    return suffixA != suffixB;
}

} // namespace


// ============================================================
// TensorRT Logger
// ============================================================

void PPQTensorRTLogger::log(
    Severity severity,
    const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout
            << "[TensorRT] "
            << msg
            << std::endl;
    }
}


// ============================================================
// Constructor
// ============================================================

PPQInt8EngineBuilder::PPQInt8EngineBuilder()
    :
    m_builder(nullptr),
    m_network(nullptr),
    m_parser(nullptr),
    m_config(nullptr)
{
    std::cout
        << "[INFO] Creating TensorRT builder..."
        << std::endl;


    m_builder =
        nvinfer1::createInferBuilder(
            m_logger
        );


    if (!m_builder)
    {
        std::cerr
            << "[ERROR] Failed to create TensorRT builder."
            << std::endl;
    }
}


// ============================================================
// Destructor
// ============================================================

PPQInt8EngineBuilder::~PPQInt8EngineBuilder()
{
    if (m_config)
    {
        delete m_config;
        m_config = nullptr;
    }


    if (m_parser)
    {
        delete m_parser;
        m_parser = nullptr;
    }


    if (m_network)
    {
        delete m_network;
        m_network = nullptr;
    }


    if (m_builder)
    {
        delete m_builder;
        m_builder = nullptr;
    }
}


// ============================================================
// Load PPQ JSON
// ============================================================

bool PPQInt8EngineBuilder::loadQuantizationInfo(
    const std::string& jsonPath)
{
    using boost::property_tree::ptree;


    ptree root;


    std::cout
        << "[INFO] Loading PPQ quantization JSON: "
        << jsonPath
        << std::endl;


    try
    {
        boost::property_tree::read_json(
            jsonPath,
            root
        );
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[ERROR] Failed to parse JSON: "
            << e.what()
            << std::endl;

        return false;
    }


    // ========================================================
    // 获取 act_quant_info
    // ========================================================

    auto actQuantInfo =
        root.get_child_optional(
            "act_quant_info"
        );


    if (!actQuantInfo)
    {
        std::cerr
            << "[ERROR] Cannot find "
            << "\"act_quant_info\" in JSON."
            << std::endl;

        return false;
    }


    m_activationQuantInfo.clear();


    // ========================================================
    // 读取 activation quantization information
    // ========================================================

    for (const auto& item : *actQuantInfo)
    {
        const std::string& tensorName =
            item.first;


        try
        {
            const float value =
                item.second.get_value<float>();


            if (!std::isfinite(value) ||
                value <= 0.0f)
            {
                std::cerr
                    << "[WARNING] Invalid quantization "
                    << "value for tensor: "
                    << tensorName
                    << " -> "
                    << value
                    << std::endl;

                continue;
            }


            m_activationQuantInfo[
                tensorName
            ] = value;
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "[WARNING] Cannot read value of tensor: "
                << tensorName
                << std::endl;
        }
    }


    std::cout
        << "[INFO] Loaded "
        << m_activationQuantInfo.size()
        << " activation quantization entries."
        << std::endl;


    if (m_activationQuantInfo.empty())
    {
        std::cerr
            << "[ERROR] No activation quantization "
            << "information found."
            << std::endl;

        return false;
    }


    return true;
}


// ============================================================
// Parse ONNX
// ============================================================

bool PPQInt8EngineBuilder::parseONNX(
    const std::string& onnxPath)
{
    if (!m_builder)
    {
        std::cerr
            << "[ERROR] TensorRT builder is invalid."
            << std::endl;

        return false;
    }


    // ========================================================
    // 创建 Network
    //
    // TensorRT 10.8
    // ========================================================

    const uint32_t networkFlags = 0U;
        


    m_network =
        m_builder->createNetworkV2(
            networkFlags
        );


    if (!m_network)
    {
        std::cerr
            << "[ERROR] Failed to create TensorRT network."
            << std::endl;

        return false;
    }


    // ========================================================
    // 创建 ONNX Parser
    // ========================================================

    m_parser =
        nvonnxparser::createParser(
            *m_network,
            m_logger
        );


    if (!m_parser)
    {
        std::cerr
            << "[ERROR] Failed to create ONNX parser."
            << std::endl;

        return false;
    }


    std::cout
        << "[INFO] Parsing ONNX: "
        << onnxPath
        << std::endl;


    const bool success =
        m_parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(
                nvinfer1::ILogger::Severity::kWARNING
            )
        );


    if (!success)
    {
        std::cerr
            << "[ERROR] ONNX parsing failed."
            << std::endl;


        for (int i = 0;
             i < m_parser->getNbErrors();
             ++i)
        {
            std::cerr
                << "[ONNX ERROR] "
                << m_parser->getError(i)->desc()
                << std::endl;
        }


        return false;
    }


    std::cout
        << "[INFO] ONNX parsing successful."
        << std::endl;


    std::cout
        << "[INFO] Network layers: "
        << m_network->getNbLayers()
        << std::endl;


    std::cout
        << "[INFO] Network inputs: "
        << m_network->getNbInputs()
        << std::endl;


    std::cout
        << "[INFO] Network outputs: "
        << m_network->getNbOutputs()
        << std::endl;


    return true;
}


// ============================================================
// Lookup dynamic range
// ============================================================

bool PPQInt8EngineBuilder::lookupDynamicRange(
    const std::string& tensorName,
    float& value) const
{
    // --------------------------------------------------------
    // 1. PPQ JSON
    // --------------------------------------------------------

    const auto jsonIt =
        m_activationQuantInfo.find(
            tensorName
        );


    if (jsonIt !=
        m_activationQuantInfo.end())
    {
        value = jsonIt->second;

        return true;
    }


    // --------------------------------------------------------
    // 2. 数据流推导结果
    // --------------------------------------------------------

    const auto derivedIt =
        m_derivedActivationInfo.find(
            tensorName
        );


    if (derivedIt !=
        m_derivedActivationInfo.end())
    {
        value = derivedIt->second;

        return true;
    }


    return false;
}


// ============================================================
// Lookup dynamic range, ignoring a trailing "_<digits>"
//
// PPQ 与 ONNX Parser 可能对同一份数据用不同的名字,例如:
//
//     JSON : /model.2/Split_output_1
//     TRT  : /model.2/Split_output
//
// 因此这里把张量名尾部的 "_<数字>" 逐层剥掉再查,
// 并且允许"同一层内取任意一个已知兄弟张量"——
// 这些都是同一份数据的视图,range 完全相同。
// ============================================================

bool PPQInt8EngineBuilder::lookupDynamicRangeByAlias(
    const std::string& tensorName,
    const std::set<std::string>& tensorsInGraph,
    float& value) const
{
    std::string base = tensorName;


    // --------------------------------------------------------
    // 逐层剥掉尾部的 "_<数字>" 再查
    //
    // 每一层都做两件事:
    //   a) 把当前名字当作完整名字查一次(带序号 ↔ 不带序号)
    //   b) 只有在当前基名确实存在于图中时,才允许
    //      跨序号借用兄弟张量的 range
    // --------------------------------------------------------

    for (int level = 0; level < 8; ++level)
    {
        // ----------------------------------------------------
        // a) 当前名字本身
        // ----------------------------------------------------

        if (lookupDynamicRange(base, value))
        {
            return true;
        }


        // ----------------------------------------------------
        // b) 兄弟序号
        //
        // 安全约束:只有 "基名" 或 "基名_<数字>" 真的存在于图中
        // 才允许借用,避免把
        //     /model.22/Constant_12_output
        // 错误地匹配到 /model.22/Constant_13_output_0。
        // ----------------------------------------------------

        bool allowSibling =
            tensorsInGraph.count(base) > 0;


        if (!allowSibling)
        {
            for (int i = 0; i < 64; ++i)
            {
                if (tensorsInGraph.count(
                        base + "_" + std::to_string(i)))
                {
                    allowSibling = true;
                    break;
                }
            }
        }


        if (allowSibling)
        {
            float best = 0.0f;

            bool found = false;


            for (int i = 0; i < 64; ++i)
            {
                float candidateRange = 0.0f;


                if (!lookupDynamicRange(
                        base + "_" + std::to_string(i),
                        candidateRange))
                {
                    continue;
                }


                best = found
                           ? std::min(best, candidateRange)
                           : candidateRange;

                found = true;
            }


            if (found)
            {
                value = best;

                return true;
            }
        }


        // ----------------------------------------------------
        // 剥掉最后一层 "_<数字>"
        // ----------------------------------------------------

        const std::size_t underscore =
            base.rfind('_');


        if (underscore == std::string::npos ||
            underscore + 1 >= base.size())
        {
            break;
        }


        const std::string suffix =
            base.substr(underscore + 1);


        bool allDigits = true;


        for (char c : suffix)
        {
            if (c < '0' || c > '9')
            {
                allDigits = false;
                break;
            }
        }


        if (!allDigits)
        {
            break;
        }


        base = base.substr(0, underscore);
    }


    return false;
}


// ============================================================
// Apply Dynamic Range
// ============================================================

bool PPQInt8EngineBuilder::applyDynamicRangeToTensor(
    nvinfer1::ITensor* tensor,
    bool derived)
{
    if (!tensor)
    {
        return false;
    }


    const char* tensorName =
        tensor->getName();


    if (!tensorName)
    {
        return false;
    }


    float dynamicRange = 0.0f;


    // --------------------------------------------------------
    // 既没有 PPQ range,也没有推导出 range
    // --------------------------------------------------------

    if (!lookupDynamicRange(
            tensorName,
            dynamicRange))
    {
        return false;
    }


    if (!std::isfinite(dynamicRange) ||
        dynamicRange <= 0.0f)
    {
        std::cerr
            << "[WARNING] Invalid dynamic range for: "
            << tensorName
            << std::endl;

        return false;
    }


    // ========================================================
    // PPQ -> TensorRT
    //
    // PPQ 的 act_quant_info 保存的就是 activation 的
    // absolute dynamic range,可以直接使用。
    //
    // TensorRT INT8 使用对称 dynamic range:
    //
    //     [-range, +range]
    // ========================================================

    tensor->setDynamicRange(
        -dynamicRange,
        dynamicRange
    );


    if (derived)
    {
        std::cout
            << "[INT8][derived] "
            << tensorName
            << " -> range ["
            << -dynamicRange
            << ", "
            << dynamicRange
            << "]"
            << std::endl;
    }
    else
    {
        std::cout
            << "[INT8] "
            << tensorName
            << " -> range ["
            << -dynamicRange
            << ", "
            << dynamicRange
            << "]"
            << std::endl;
    }


    return true;
}


// ============================================================
// Infer one missing dynamic range
// ============================================================

bool PPQInt8EngineBuilder::inferMissingRange(
    nvinfer1::ITensor* tensor,
    int& rule)
{
    rule = 0;


    if (!tensor ||
        !m_network)
    {
        return false;
    }


    const char* rawTensorName =
        tensor->getName();


    if (!rawTensorName)
    {
        return false;
    }


    const std::string tensorName =
        rawTensorName;


    // --------------------------------------------------------
    // 已经有 range 了
    // --------------------------------------------------------

    float existing = 0.0f;


    if (lookupDynamicRange(
            tensorName,
            existing))
    {
        return true;
    }


    // --------------------------------------------------------
    // 精确名匹配不上时,先试别名
    //
    // 覆盖 "JSON 与 TensorRT 命名不一致" 的情况:
    //
    //     TRT  : /model.2/Split_output
    //     JSON : /model.2/Split_output_1
    //
    // tensorsInGraph 用于防止跨层误匹配,只构建一次。
    // --------------------------------------------------------

    float aliasRange = 0.0f;


    if (!m_graphTensorNamesBuilt)
    {
        m_graphTensorNames.clear();


        for (int i = 0;
             i < m_network->getNbLayers();
             ++i)
        {
            nvinfer1::ILayer* layer =
                m_network->getLayer(i);


            if (!layer)
            {
                continue;
            }


            for (int j = 0;
                 j < layer->getNbOutputs();
                 ++j)
            {
                nvinfer1::ITensor* output =
                    layer->getOutput(j);


                if (!output)
                {
                    continue;
                }


                const char* outputName =
                    output->getName();


                if (outputName)
                {
                    m_graphTensorNames.insert(outputName);
                }
            }
        }


        m_graphTensorNamesBuilt = true;
    }


    if (lookupDynamicRangeByAlias(
            tensorName,
            m_graphTensorNames,
            aliasRange))
    {
        m_derivedActivationInfo[
            tensorName
        ] = aliasRange;

        m_rangeOrigin[
            tensorName
        ] = 1;

        rule = 1;

        return true;
    }


    // ========================================================
    // 找到产生这个张量的 Layer,以及消费这个张量的 Layer
    //
    // NOTE:
    //   这里通过"张量名字"比较,而不是比较 ITensor*。
    //   原因是 ONNX Parser 有时会为同一个张量返回不同的
    //   ITensor 包装对象,按名字比较更可靠。
    // ========================================================

    nvinfer1::ILayer* producer = nullptr;

    int producerOutputIndex = -1;


    for (int i = 0;
         i < m_network->getNbLayers() && !producer;
         ++i)
    {
        nvinfer1::ILayer* layer =
            m_network->getLayer(i);


        if (!layer)
        {
            continue;
        }


        for (int j = 0;
             j < layer->getNbOutputs();
             ++j)
        {
            nvinfer1::ITensor* candidate =
                layer->getOutput(j);


            if (!candidate)
            {
                continue;
            }


            const char* candidateName =
                candidate->getName();


            if (candidateName &&
                tensorName == candidateName)
            {
                producer = layer;
                producerOutputIndex = j;
                break;
            }
        }
    }


    std::vector<nvinfer1::ILayer*> consumers;


    for (int i = 0;
         i < m_network->getNbLayers();
         ++i)
    {
        nvinfer1::ILayer* layer =
            m_network->getLayer(i);


        if (!layer)
        {
            continue;
        }


        for (int j = 0;
             j < layer->getNbInputs();
             ++j)
        {
            nvinfer1::ITensor* candidate =
                layer->getInput(j);


            if (!candidate)
            {
                continue;
            }


            const char* candidateName =
                candidate->getName();


            if (candidateName &&
                tensorName == candidateName)
            {
                consumers.push_back(layer);
                break;
            }
        }
    }


    // --------------------------------------------------------
    // 工具:记录一个推导出来的 range
    // --------------------------------------------------------

    int firedRule = 0;


    auto record =
        [&](float value) -> bool
    {
        if (!std::isfinite(value) ||
            value <= 0.0f)
        {
            return false;
        }


        m_derivedActivationInfo[
            tensorName
        ] = value;

        m_rangeOrigin[
            tensorName
        ] = firedRule;

        rule = firedRule;


        return true;
    };


    // ========================================================
    // 规则 1: 命名同源的张量
    //
    // PPQ / ONNX 的命名规则:
    //
    //     /model.2/Split_output_0
    //     /model.2/Split_output_1
    //
    // 多个 Split 输出是同一份数据的不同视图,数值完全相同。
    // PPQ 有时只记录其中一个(通常是 _1),另一个就缺失了。
    //
    // 这里直接用命名规则兜底,不依赖 ONNX Parser 把 Split
    // 映射成哪一种 LayerType。
    //
    // 例:
    //     /model.2/Split_output_0  (JSON 里没有)
    //         <- /model.2/Split_output_1  (JSON 里有)
    // ========================================================

    firedRule = 2;

    for (int i = 0;
         i < m_network->getNbLayers();
         ++i)
    {
        nvinfer1::ILayer* layer =
            m_network->getLayer(i);


        if (!layer)
        {
            continue;
        }


        for (int j = 0;
             j < layer->getNbOutputs();
             ++j)
        {
            nvinfer1::ITensor* candidate =
                layer->getOutput(j);


            if (!candidate)
            {
                continue;
            }


            const char* candidateName =
                candidate->getName();


            if (!candidateName)
            {
                continue;
            }


            if (!isIndexedSibling(
                    tensorName,
                    candidateName))
            {
                continue;
            }


            float siblingRange = 0.0f;


            if (lookupDynamicRange(
                    candidateName,
                    siblingRange) &&
                record(siblingRange))
            {
                return true;
            }
        }
    }


    // ========================================================
    // 规则 2: Layer 的其他输出
    //
    // 保值算子(Split / Slice ...)的多个输出值相同。
    // 这是规则 1 之外的兜底,覆盖命名不规则的情况。
    // ========================================================

    firedRule = 3;

    if (producer &&
        isValuePreservingLayer(
            producer->getType()))
    {
        for (int j = 0;
             j < producer->getNbOutputs();
             ++j)
        {
            if (j == producerOutputIndex)
            {
                continue;
            }


            nvinfer1::ITensor* sibling =
                producer->getOutput(j);


            if (!sibling)
            {
                continue;
            }


            const char* siblingName =
                sibling->getName();


            if (!siblingName)
            {
                continue;
            }


            float siblingRange = 0.0f;


            if (lookupDynamicRange(
                    siblingName,
                    siblingRange) &&
                record(siblingRange))
            {
                return true;
            }
        }
    }


    // ========================================================
    // 规则 3: 保值算子的输入
    //
    // Reshape / Slice / Transpose / Split ...
    // 输出与输入数值相同。
    //
    // 例:
    //     /model.22/Reshape_output_0
    //         <- /model.22/cv2.0/cv2.0.2/Conv_output_0
    // ========================================================

    firedRule = 4;

    if (producer &&
        isValuePreservingLayer(
            producer->getType()))
    {
        for (int j = 0;
             j < producer->getNbInputs();
             ++j)
        {
            nvinfer1::ITensor* input =
                producer->getInput(j);


            if (!input)
            {
                continue;
            }


            const char* inputName =
                input->getName();


            if (!inputName)
            {
                continue;
            }


            float inputRange = 0.0f;


            if (lookupDynamicRange(
                    inputName,
                    inputRange) &&
                record(inputRange))
            {
                return true;
            }
        }
    }


    // ========================================================
    // 规则 4: 有界算子
    //
    // Sigmoid / Softmax / MaxPool ...
    // 输出取值一定落在输入范围之内,
    // 所以直接用输入 range 作为输出 range 是安全上界。
    // ========================================================

    firedRule = 5;

    if (producer &&
        isBoundedRangeLayer(
            producer->getType()))
    {
        for (int j = 0;
             j < producer->getNbInputs();
             ++j)
        {
            nvinfer1::ITensor* input =
                producer->getInput(j);


            if (!input)
            {
                continue;
            }


            const char* inputName =
                input->getName();


            if (!inputName)
            {
                continue;
            }


            float inputRange = 0.0f;


            if (lookupDynamicRange(
                    inputName,
                    inputRange) &&
                record(inputRange))
            {
                return true;
            }
        }
    }


    // ========================================================
    // 规则 5: 并集算子
    //
    // Concat / Add / Mul ...
    // 取所有已知输入 range 的最大值,是安全上界。
    // ========================================================

    firedRule = 6;

    if (producer &&
        isUnionRangeLayer(
            producer->getType()))
    {
        float maxRange = 0.0f;

        bool found = false;


        for (int j = 0;
             j < producer->getNbInputs();
             ++j)
        {
            nvinfer1::ITensor* input =
                producer->getInput(j);


            if (!input)
            {
                continue;
            }


            const char* inputName =
                input->getName();


            if (!inputName)
            {
                continue;
            }


            float inputRange = 0.0f;


            if (lookupDynamicRange(
                    inputName,
                    inputRange))
            {
                maxRange =
                    found
                        ? std::max(
                              maxRange,
                              inputRange)
                        : inputRange;

                found = true;
            }
        }


        if (found && record(maxRange))
        {
            return true;
        }
    }


    // ========================================================
    // 规则 6: 反向推导
    //
    // 消费这个张量的算子如果是保值算子,那么
    // range(output) == range(input) == range(这个张量)。
    //
    // 例:
    //     /model.22/Concat_output_0
    //         -> /model.22/dfl/Reshape_output_0 (JSON 里有)
    // ========================================================

    firedRule = 7;

    for (nvinfer1::ILayer* consumer : consumers)
    {
        if (!consumer ||
            !isValuePreservingLayer(
                consumer->getType()))
        {
            continue;
        }


        float minRadius =
            std::numeric_limits<float>::max();

        bool found = false;


        for (int j = 0;
             j < consumer->getNbOutputs();
             ++j)
        {
            nvinfer1::ITensor* output =
                consumer->getOutput(j);


            if (!output)
            {
                continue;
            }


            const char* outputName =
                output->getName();


            if (!outputName)
            {
                continue;
            }


            float outputRange = 0.0f;


            if (lookupDynamicRange(
                    outputName,
                    outputRange))
            {
                minRadius =
                    found
                        ? std::min(
                              minRadius,
                              outputRange)
                        : outputRange;

                found = true;
            }
        }


        if (found && record(minRadius))
        {
            return true;
        }
    }


    return false;
}


// ============================================================
// Propagate dynamic ranges until fixpoint
// ============================================================

int PPQInt8EngineBuilder::propagateDynamicRanges()
{
    if (!m_network)
    {
        return 0;
    }


    // --------------------------------------------------------
    // 收集需要补 range 的所有张量
    //
    // 顺序:
    //   1. Network Inputs
    //   2. 所有 Layer Outputs
    //   3. Network Outputs
    // --------------------------------------------------------

    std::vector<nvinfer1::ITensor*> tensors;


    for (int i = 0;
         i < m_network->getNbInputs();
         ++i)
    {
        if (nvinfer1::ITensor* tensor =
                m_network->getInput(i))
        {
            tensors.push_back(tensor);
        }
    }


    for (int i = 0;
         i < m_network->getNbLayers();
         ++i)
    {
        nvinfer1::ILayer* layer =
            m_network->getLayer(i);


        if (!layer)
        {
            continue;
        }


        for (int j = 0;
             j < layer->getNbOutputs();
             ++j)
        {
            if (nvinfer1::ITensor* tensor =
                    layer->getOutput(j))
            {
                tensors.push_back(tensor);
            }
        }
    }


    for (int i = 0;
         i < m_network->getNbOutputs();
         ++i)
    {
        if (nvinfer1::ITensor* tensor =
                m_network->getOutput(i))
        {
            tensors.push_back(tensor);
        }
    }


    // --------------------------------------------------------
    // 迭代到 fixpoint
    //
    // 每一轮只做"已知 -> 未知"的一步传播,
    // 因此最长链上的张量需要多轮才能补全。
    // 收敛条件是某一轮没有任何新增。
    // --------------------------------------------------------

    int totalDerived = 0;


    const int maxRounds =
        m_network->getNbLayers() + 1;


    for (int round = 0;
         round < maxRounds;
         ++round)
    {
        int addedThisRound = 0;


        for (nvinfer1::ITensor* tensor : tensors)
        {
            const char* name =
                tensor->getName();


            if (!name)
            {
                continue;
            }


            float existing = 0.0f;


            if (lookupDynamicRange(
                    name,
                    existing))
            {
                continue;
            }


            int rule = 0;


            if (inferMissingRange(tensor, rule))
            {
                addedThisRound++;
            }
        }


        if (addedThisRound == 0)
        {
            break;
        }


        totalDerived += addedThisRound;
    }


    return totalDerived;
}


// ============================================================
// Apply all activation dynamic ranges
// ============================================================

int PPQInt8EngineBuilder::applyActivationDynamicRanges()
{
    if (!m_network)
    {
        std::cerr
            << "[ERROR] TensorRT network is invalid."
            << std::endl;

        return 0;
    }


    const std::size_t jsonCount =
        m_activationQuantInfo.size();


    std::cout
        << "\n[INFO] PPQ JSON activation entries: "
        << jsonCount
        << std::endl;


    // ========================================================
    // Step 1
    // 用数据流推导补齐 JSON 未覆盖的张量
    // ========================================================

    std::cout
        << "[INFO] Inferring missing dynamic ranges "
        << "from the network dataflow..."
        << std::endl;


    const int derivedCount =
        propagateDynamicRanges();


    std::cout
        << "[INFO] Inferred dynamic ranges for "
        << derivedCount
        << " extra tensors."
        << std::endl;


    // --------------------------------------------------------
    // 按来源统计
    //
    //   direct = 精确命中 PPQ JSON 的条目数
    //   其余   = m_rangeOrigin 中的别名 / 推导 / 兜底
    // --------------------------------------------------------

    int originCount[4] = { 0, 0, 0, 0 };


    originCount[0] = static_cast<int>(
        m_activationQuantInfo.size()
    );


    for (const auto& item : m_rangeOrigin)
    {
        const int origin = item.second;


        if (origin >= 0 && origin <= 3)
        {
            originCount[origin]++;
        }
    }


    std::cout
        << "[INFO] Range origin: direct-json="
        << originCount[0]
        << ", alias="
        << originCount[1]
        << ", dataflow="
        << originCount[2]
        << ", fallback="
        << originCount[3]
        << std::endl;


    // ========================================================
    // Step 2
    // 逐一写入 TensorRT Network
    //
    // 同一个 ITensor 既可能出现在 Layer Outputs 里,
    // 也可能同时是 Network Output(例如 output0),
    // 因此这里先按张量名去重,避免重复统计。
    // ========================================================

    std::vector<nvinfer1::ITensor*> tensors;


    for (int i = 0;
         i < m_network->getNbInputs();
         ++i)
    {
        if (nvinfer1::ITensor* tensor =
                m_network->getInput(i))
        {
            tensors.push_back(tensor);
        }
    }


    for (int i = 0;
         i < m_network->getNbLayers();
         ++i)
    {
        nvinfer1::ILayer* layer =
            m_network->getLayer(i);


        if (!layer)
        {
            continue;
        }


        for (int j = 0;
             j < layer->getNbOutputs();
             ++j)
        {
            if (nvinfer1::ITensor* tensor =
                    layer->getOutput(j))
            {
                tensors.push_back(tensor);
            }
        }
    }


    for (int i = 0;
         i < m_network->getNbOutputs();
         ++i)
    {
        if (nvinfer1::ITensor* tensor =
                m_network->getOutput(i))
        {
            tensors.push_back(tensor);
        }
    }


    int appliedCount = 0;


    std::set<std::string> appliedNames;

    std::vector<std::string> missingNames;

    std::set<std::string> missingSeen;


    for (nvinfer1::ITensor* tensor : tensors)
    {
        if (!tensor)
        {
            continue;
        }


        const char* rawName =
            tensor->getName();


        if (!rawName)
        {
            continue;
        }


        const std::string name =
            rawName;


        // ----------------------------------------------------
        // 同一个张量只处理一次
        // ----------------------------------------------------

        if (appliedNames.count(name))
        {
            continue;
        }


        appliedNames.insert(name);


        // ----------------------------------------------------
        // 精确命中 PPQ JSON 的算 "direct",其余都算推导得到
        // ----------------------------------------------------

        const bool exactJson =
            m_activationQuantInfo.find(name) !=
            m_activationQuantInfo.end();


        if (applyDynamicRangeToTensor(
                tensor,
                !exactJson))
        {
            appliedCount++;
            continue;
        }


        // ----------------------------------------------------
        // 这个张量没有 range 可用,留给兜底阶段
        // ----------------------------------------------------

        if (!missingSeen.count(name))
        {
            missingSeen.insert(name);
            missingNames.push_back(name);
        }
    }


    // ========================================================
    // Step 2.5
    // 兜底:保证每个张量都有 dynamic range
    //
    // TensorRT 只需要"每个非 INT32 张量都有 range"。
    // 到这里还缺 range 的张量说明 PPQ JSON 与当前 ONNX
    // 命名对不上(例如 JSON 来自另一次导出)。
    //
    // 此时宁可给一个保守的大上界,也不要留下缺失——
    // 缺失只会让 TensorRT 打印 "Missing scale and zero-point"
    // 并回退到 FP32 实现。
    //
    // 上界取 JSON / 推导结果中的最大值,并向上取整到 2 的幂,
    // 保证不会比任何已知张量更紧。
    // ========================================================

    if (!missingNames.empty())
    {
        float globalMax = 1.0f;


        for (const auto& item : m_activationQuantInfo)
        {
            if (std::isfinite(item.second) &&
                item.second > globalMax)
            {
                globalMax = item.second;
            }
        }


        for (const auto& item : m_derivedActivationInfo)
        {
            if (std::isfinite(item.second) &&
                item.second > globalMax)
            {
                globalMax = item.second;
            }
        }


        float fallbackRange = 1.0f;


        while (fallbackRange < globalMax &&
               fallbackRange < 1.0e30f)
        {
            fallbackRange *= 2.0f;
        }


        std::cerr
            << "\n[WARNING] "
            << missingNames.size()
            << " tensor(s) have no range in the PPQ JSON and could "
            << "not be matched by name or inferred from the dataflow."
            << std::endl;

        std::cerr
            << "[WARNING] A few of these (Shape / Gather shape chains, "
            << "elementwise layers) are normal and harmless."
            << std::endl;

        std::cerr
            << "[WARNING] A large number means the JSON was exported "
            << "from a different graph than the ONNX: their ranges are "
            << "now set to the global upper bound "
            << fallbackRange
            << ", which will hurt accuracy. Re-export the ONNX and JSON "
            << "from the same run to avoid this."
            << std::endl;


        for (const std::string& name : missingNames)
        {
            m_derivedActivationInfo[
                name
            ] = fallbackRange;

            m_rangeOrigin[
                name
            ] = 3;
        }


        // ----------------------------------------------------
        // 重新应用一遍
        // ----------------------------------------------------

        std::vector<std::string> stillMissing;

        std::set<std::string> stillMissingSeen;


        for (nvinfer1::ITensor* tensor : tensors)
        {
            if (!tensor)
            {
                continue;
            }


            const char* rawName =
                tensor->getName();


            if (!rawName)
            {
                continue;
            }


            const std::string name =
                rawName;


            const auto originIt =
                m_rangeOrigin.find(name);


            if (originIt == m_rangeOrigin.end() ||
                originIt->second != 3)
            {
                continue;
            }


            if (applyDynamicRangeToTensor(
                    tensor,
                    true))
            {
                appliedCount++;
            }
            else if (!stillMissingSeen.count(name))
            {
                stillMissingSeen.insert(name);
                stillMissing.push_back(name);
            }
        }


        missingNames.swap(stillMissing);
    }


    std::cout
        << "\n[INFO] Applied dynamic ranges to "
        << appliedCount
        << " tensors"
        << "  (json="
        << jsonCount
        << ", inferred="
        << m_derivedActivationInfo.size()
        << ")"
        << std::endl;


    // ========================================================
    // Step 3
    // 报告最终仍然缺少 dynamic range 的张量
    // ========================================================

    if (missingNames.empty())
    {
        std::cout
            << "[INFO] Every tensor TensorRT enumerates now has "
            << "a dynamic range."
            << std::endl;
    }
    else
    {
        std::cerr
            << "\n[WARNING] "
            << missingNames.size()
            << " tensor(s) still have no dynamic range. "
            << "TensorRT will fall back to a non-INT8 "
            << "implementation for the layers using them:"
            << std::endl;


        for (const std::string& name : missingNames)
        {
            std::cerr
                << "    - "
                << name
                << std::endl;
        }


        std::cerr
            << "[WARNING] These should be INT32 shape / index tensors "
            << "(Shape / Gather on a shape) or elementwise layers whose "
            << "range TensorRT derives from their inputs."
            << std::endl;

        std::cerr
            << "[WARNING] If TensorRT still reports "
            << "\"Missing scale and zero-point\" for one of them, "
            << "please share that log."
            << std::endl;
    }


    return appliedCount;
}



// ============================================================
// Save Engine
// ============================================================

bool PPQInt8EngineBuilder::saveEngine(
    nvinfer1::IHostMemory* serializedEngine,
    const std::string& enginePath)
{
    if (!serializedEngine)
    {
        std::cerr
            << "[ERROR] Serialized engine is null."
            << std::endl;

        return false;
    }


    std::ofstream file(
        enginePath,
        std::ios::binary
    );


    if (!file)
    {
        std::cerr
            << "[ERROR] Cannot open output engine: "
            << enginePath
            << std::endl;

        return false;
    }


    file.write(
        static_cast<const char*>(
            serializedEngine->data()
        ),
        serializedEngine->size()
    );


    file.close();


    std::cout
        << "[INFO] Engine saved to: "
        << enginePath
        << std::endl;


    std::cout
        << "[INFO] Engine size: "
        << serializedEngine->size()
        << " bytes"
        << std::endl;


    return true;
}


// ============================================================
// Build Engine
// ============================================================

bool PPQInt8EngineBuilder::buildEngine(
    const std::string& enginePath)
{
    if (!m_builder ||
        !m_network)
    {
        std::cerr
            << "[ERROR] Builder or network is invalid."
            << std::endl;

        return false;
    }


    // ========================================================
    // 创建 Builder Config
    // ========================================================

    m_config =
        m_builder->createBuilderConfig();


    if (!m_config)
    {
        std::cerr
            << "[ERROR] Failed to create BuilderConfig."
            << std::endl;

        return false;
    }


    // ========================================================
    // INT8
    // ========================================================

    m_config->setFlag(
        nvinfer1::BuilderFlag::kINT8
    );


    std::cout
        << "[INFO] TensorRT INT8 mode enabled."
        << std::endl;


    // ========================================================
    // Workspace
    //
    // TensorRT 10.x:
    // setMemoryPoolLimit()
    // ========================================================

    m_config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        4ULL * 1024ULL * 1024ULL * 1024ULL
    );


    // ========================================================
    // Build Serialized Network
    // ========================================================

    std::cout
        << "\n[INFO] Building TensorRT INT8 engine..."
        << std::endl;


    nvinfer1::IHostMemory* serializedEngine =
        m_builder->buildSerializedNetwork(
            *m_network,
            *m_config
        );


    if (!serializedEngine)
    {
        std::cerr
            << "[ERROR] TensorRT engine build failed."
            << std::endl;

        return false;
    }


    // ========================================================
    // Save
    // ========================================================

    const bool success =
        saveEngine(
            serializedEngine,
            enginePath
        );


    delete serializedEngine;


    return success;
}


// ============================================================
// Complete Build
// ============================================================

bool PPQInt8EngineBuilder::build(
    const std::string& onnxPath,
    const std::string& jsonPath,
    const std::string& enginePath)
{
    std::cout
        << "============================================"
        << std::endl;

    std::cout
        << " PPQ -> TensorRT INT8 Engine"
        << std::endl;

    std::cout
        << " TensorRT 10.x"
        << std::endl;

    std::cout
        << "============================================"
        << std::endl;


    // ========================================================
    // Step 1
    // JSON
    // ========================================================

    if (!loadQuantizationInfo(
            jsonPath))
    {
        return false;
    }


    // ========================================================
    // Step 2
    // ONNX
    // ========================================================

    if (!parseONNX(
            onnxPath))
    {
        return false;
    }


    // ========================================================
    // Step 3
    // Apply PPQ activation ranges
    // ========================================================

    const int appliedCount =
        applyActivationDynamicRanges();


    if (appliedCount == 0)
    {
        std::cerr
            << "[ERROR] No activation dynamic ranges "
            << "were applied."
            << std::endl;

        return false;
    }


    // ========================================================
    // Step 4
    // Build Engine
    // ========================================================

    if (!buildEngine(
            enginePath))
    {
        return false;
    }


    std::cout
        << "\n============================================"
        << std::endl;

    std::cout
        << " BUILD SUCCESS"
        << std::endl;

    std::cout
        << "============================================"
        << std::endl;


    return true;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <onnx> <json> <engine>\n\n"

            << "Example:\n"
            << "  "
            << argv[0]
            << " yolov8n_int8_ppq.onnx "
            << "yolov8n_int8_ppq.json "
            << "yolov8n_int8_ppq.engine\n";

        return -1;
    }


    const std::string onnxPath =
        argv[1];

    const std::string jsonPath =
        argv[2];

    const std::string enginePath =
        argv[3];


    PPQInt8EngineBuilder builder;


    if (!builder.build(
            onnxPath,
            jsonPath,
            enginePath))
    {
        std::cerr
            << "\n[ERROR] INT8 engine build failed."
            << std::endl;

        return -1;
    }


    return 0;
}