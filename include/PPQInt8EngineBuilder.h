#ifndef PPQ_INT8_ENGINE_BUILDER_H
#define PPQ_INT8_ENGINE_BUILDER_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <string>
#include <unordered_map>
#include <set>
#include <vector>


/**
 * @brief TensorRT Logger
 */
class PPQTensorRTLogger : public nvinfer1::ILogger
{
public:

    void log(
        Severity severity,
        const char* msg) noexcept override;
};


/**
 * @brief
 * 使用 PPQ 导出的 activation quantization information
 * 将 ONNX 模型构建为 TensorRT INT8 Engine。
 *
 * 输入：
 *   1. PPQ INT8 ONNX
 *   2. PPQ quantization JSON
 *
 * 输出：
 *   TensorRT INT8 Engine
 *
 * 当前版本：
 *   - 只使用 act_quant_info
 *   - 权重由 TensorRT 自行处理
 *   - 不使用 TensorRT Calibrator
 *   - 暂不进行 INT8/FP16 混合精度控制
 *
 * 关于 dynamic range 的缺失：
 *
 *   PPQ 的 act_quant_info 并不保证覆盖 ONNX 图中的每一个张量。
 *   例如 Split_output_0 / Reshape_output_0 这类"纯搬运数据"的算子
 *   经常没有出现在 JSON 里，但 TensorRT 在构建 INT8 engine 时
 *   仍然要求这些张量具备 dynamic range，否则会打印：
 *
 *     [TensorRT] Missing scale and zero-point for tensor ...
 *     ... fall back to non-int8 implementation ...
 *
 *   因此本类在应用 JSON 中的 range 之后，会对剩余张量做一次
 *   数据流推导(inference)，把 range 从"已知张量"传播到"未知张量"，
 *   从而消除上述 fallback。
 */
class PPQInt8EngineBuilder
{
public:

    PPQInt8EngineBuilder();

    ~PPQInt8EngineBuilder();


    /**
     * @brief
     * 从 PPQ JSON 中加载 activation quantization information
     *
     * JSON 格式：
     *
     * {
     *     "act_quant_info": {
     *         "images": 0.9346203124,
     *         "/model.0/conv/Conv_output_0": 23.90396009,
     *         ...
     *     }
     * }
     */
    bool loadQuantizationInfo(
        const std::string& jsonPath);


    /**
     * @brief
     * 解析 PPQ 导出的 ONNX
     */
    bool parseONNX(
        const std::string& onnxPath);


    /**
     * @brief
     * 将 PPQ activation quantization information
     * 应用到 TensorRT Network。
     *
     * 分成两步：
     *   1. 直接使用 JSON 中已有的 range
     *   2. 对 JSON 未覆盖的张量，通过数据流推导出 range
     *
     * @return 成功设置了 dynamic range 的张量数量
     */
    int applyActivationDynamicRanges();


    /**
     * @brief
     * 构建 TensorRT INT8 Engine 并保存。
     */
    bool buildEngine(
        const std::string& enginePath);


    /**
     * @brief
     * 一步完成：
     *
     * JSON
     *   ↓
     * ONNX
     *   ↓
     * Dynamic Range
     *   ↓
     * INT8 Engine
     */
    bool build(
        const std::string& onnxPath,
        const std::string& jsonPath,
        const std::string& enginePath);


private:

    /**
     * @brief
     * 查询某个张量的 dynamic range。
     *
     * 先查 PPQ JSON,再查推导结果。
     *
     * @param tensorName 张量名
     * @param value      输出参数,查到时写入 range
     *
     * @return 查到时返回 true
     */
    bool lookupDynamicRange(
        const std::string& tensorName,
        float& value) const;


    /**
     * @brief
     * 忽略尾部的 "_<数字>" 再查一次。
     *
     * PPQ JSON 与 ONNX Parser 对同一份数据的命名可能不一致:
     *
     *     JSON :  /model.2/Split_output_1
     *     TRT  :  /model.2/Split_output
     *
     * 因此当精确名匹配不上时,把张量名尾部连续的 "_<数字>"
     * 逐层剥掉再查,并且允许"同一层内取一个已知兄弟张量"。
     *
     * @param tensorName    张量名
     * @param tensorsInGraph 网络中真实存在的全部张量名,
     *                       用于防止跨层误匹配
     * @param value         输出参数,查到时写入 range
     *
     * @return 查到时返回 true
     */
    bool lookupDynamicRangeByAlias(
        const std::string& tensorName,
        const std::set<std::string>& tensorsInGraph,
        float& value) const;


    /**
     * @brief
     * 为单个 Tensor 设置 dynamic range。
     *
     * @param derived 该 range 是否不是精确命中 JSON
     *                (true  = 由别名匹配 / 数据流推导 / 全局兜底得到)
     *                (false = 直接来自 PPQ JSON)
     */
    bool applyDynamicRangeToTensor(
        nvinfer1::ITensor* tensor,
        bool derived);


    /**
     * @brief
     * 推导单个张量的 dynamic range(仅查当前已知信息)。
     *
     * 推导顺序:
     *   1. 去掉尾部 "_<数字>" 的别名匹配
     *      (/model.2/Split_output <- /model.2/Split_output_1)
     *   2. 命名同源的张量
     *      (/model.2/Split_output_0 <- /model.2/Split_output_1)
     *   3. 该张量所属 Layer 的其他输出
     *   4. 该张量所属 Layer 的输入(Reshape / Slice ... 等保值算子)
     *   5. 有界算子(Sigmoid / Softmax / MaxPool)的输入
     *   6. 并集算子(Concat / Add / Mul ...)的所有输入取最大值
     *   7. 消费该张量的保值算子的输出(反向推导)
     *
     * @param tensor 待推导的张量
     * @param rule   输出参数,成功时写入命中的规则编号(1..7)
     *
     * @return 成功推导出 range 时返回 true
     */
    bool inferMissingRange(
        nvinfer1::ITensor* tensor,
        int& rule);


    /**
     * @brief
     * 迭代执行 inferMissingRange,直到没有新的 range 可以推导出来。
     *
     * @return 新增的 range 数量
     */
    int propagateDynamicRanges();


    /**
     * @brief 保存 TensorRT Serialized Engine。
     */
    bool saveEngine(
        nvinfer1::IHostMemory* serializedEngine,
        const std::string& enginePath);


private:

    PPQTensorRTLogger m_logger;

    nvinfer1::IBuilder* m_builder;

    nvinfer1::INetworkDefinition* m_network;

    nvonnxparser::IParser* m_parser;

    nvinfer1::IBuilderConfig* m_config;


    /**
     * PPQ activation quantization information
     *
     * key:
     *     Tensor name
     *
     * value:
     *     activation dynamic range
     */
    std::unordered_map<std::string, float>
        m_activationQuantInfo;


    /**
     * 由数据流推导得到的 activation dynamic range
     *
     * key:
     *     Tensor name
     *
     * value:
     *     推导出的 activation dynamic range
     */
    std::unordered_map<std::string, float>
        m_derivedActivationInfo;


    /**
     * 某条 range 的来源,用于最终报告
     *
     * key:
     *     Tensor name
     *
     * value:
     *     0 = 精确来自 PPQ JSON
     *     1 = 去掉尾部 "_<数字>" 的别名匹配
     *     2 = 数据流推导
     *     3 = 全局兜底上界
     */
    std::unordered_map<std::string, int>
        m_rangeOrigin;


    /**
     * 网络中真实存在的全部张量名(懒加载缓存)
     *
     * 用于别名匹配时判断"这个名字到底存不存在",
     * 避免跨层误匹配。
     */
    std::set<std::string>
        m_graphTensorNames;

    bool m_graphTensorNamesBuilt = false;
};


#endif // PPQ_INT8_ENGINE_BUILDER_H