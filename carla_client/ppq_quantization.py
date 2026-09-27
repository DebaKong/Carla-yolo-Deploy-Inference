import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import cv2
import torch
import numpy as np

from ppq import (
    TargetPlatform,
    graphwise_error_analyse,
    layerwise_error_analyse,
    TorchExecutor,
)

from ppq.api import (
    load_onnx_graph,
    export_ppq_graph,
    QuantizationSettingFactory,
)

import ppq.lib as PFL

from ppq.quantization.optim import (
    QuantizeSimplifyPass,
    QuantizeFusionPass,
    ParameterQuantizePass,
    RuntimeCalibrationPass,
    PassiveParameterQuantizePass,
    QuantAlignmentPass,
)


# ============================================================
# 配置
# ============================================================

ONNX_PATH = "./ppq_output/yolov8n.onnx"

CALIB_DIR = "./Calibration/images"

OUTPUT_DIR = "./ppq_output"

INPUT_SHAPE = [1, 3, 640, 640]

DEVICE = "cpu"

CALIB_STEPS = 379


# ============================================================
# 数据预处理
# ============================================================

def preprocess(path):

    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"读取失败: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (640, 640))

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, 0)

    return torch.from_numpy(img)


# ============================================================
# Calibration Dataset
# ============================================================

def load_calibration_data():

    files = []

    for ext in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]:

        files += glob.glob(
            os.path.join(
                CALIB_DIR,
                "**",
                ext
            ),
            recursive=True
        )

    files = sorted(set(files))[:CALIB_STEPS]

    if not files:
        raise RuntimeError("校准集为空")

    print(
        f"Calibration images: {len(files)}"
    )

    return [
        preprocess(x)
        for x in files
    ]


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):

    return batch.to(DEVICE)


# ============================================================
# Graphwise Error
# ============================================================

def graphwise_analysis(
    graph,
    dataset
):

    print(
        "\n========== Graphwise SNR =========="
    )

    result = graphwise_error_analyse(
        graph=graph,
        running_device=DEVICE,
        method="snr",
        steps=len(dataset),
        dataloader=dataset,
        collate_fn=collate_fn
    )

    for name, value in result.items():

        print(
            f"{name:<50} {value}"
        )


# ============================================================
# Layerwise Error
# ============================================================

def layerwise_analysis(
    graph,
    dataset
):

    print(
        "\n========== Layerwise Error =========="
    )

    result = layerwise_error_analyse(
        graph=graph,
        running_device=DEVICE,
        dataloader=dataset,
        collate_fn=collate_fn
    )

    for name, value in result.items():

        print(
            f"{name:<50} {value}"
        )


# ============================================================
# Main
# ============================================================

def main():

    if not ONNX_PATH:

        raise ValueError(
            "请填写 ONNX_PATH"
        )

    if not CALIB_DIR:

        raise ValueError(
            "请填写 CALIB_DIR"
        )

    if DEVICE == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA 不可用"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # Calibration Dataset
    # ========================================================

    dataset = load_calibration_data()


    # ========================================================
    # Load ONNX
    # ========================================================

    print(
        "\nLoading ONNX..."
    )

    graph = load_onnx_graph(
        onnx_import_file=ONNX_PATH
    )

    print(
        f"Operations before quantization: "
        f"{len(graph.operations)}"
    )


    # ========================================================
    # PPQ Quantizer
    # ========================================================

    print(
        "\nCreating PPQ TensorRT INT8 quantizer..."
    )

    quantizer = PFL.Quantizer(

        platform=TargetPlatform.TRT_INT8,

        graph=graph
    )


    # ========================================================
    # Dispatch
    # ========================================================

    dispatching = PFL.Dispatcher(
        graph=graph
    ).dispatch(
        quant_types=quantizer.quant_operation_types
    )


    # ========================================================
    # Initialize quantization information
    # ========================================================

    print(
        "\nInitializing quantization information..."
    )

    for op in graph.operations.values():

        quantizer.quantize_operation(

            op_name=op.name,

            platform=dispatching[op.name]
        )


    # ========================================================
    # Executor
    # ========================================================

    executor = TorchExecutor(

        graph=graph,

        device=DEVICE
    )


    executor.tracing_operation_meta(
        inputs=torch.zeros(
            INPUT_SHAPE
        )
    )

    executor.load_graph(
        graph=graph
    )


    # ========================================================
    # PPQ Official TensorRT Optimization Pipeline
    # ========================================================

    print(
        "\nCreating PPQ TensorRT optimization pipeline..."
    )

    pipeline = PFL.Pipeline([

        # 1. 删除/合并重复的量化配置
        QuantizeSimplifyPass(),

        # 2. TensorRT 相关量化图融合
        QuantizeFusionPass(
            activation_type=
            quantizer.activation_fusion_types
        ),

        # 3. 权重量化
        ParameterQuantizePass(),

        # 4. 使用 calibration dataset
        #    计算 activation scale
        RuntimeCalibrationPass(),

        # 5. 被动参数量化
        PassiveParameterQuantizePass(),

        # 6. TensorRT 量化参数对齐
        QuantAlignmentPass(
            force_overlap=True
        ),

    ])


    # ========================================================
    # Execute Quantization Pipeline
    # ========================================================

    print(
        "\nStarting PPQ TensorRT INT8 quantization..."
    )

    pipeline.optimize(

        graph=graph,

        dataloader=dataset,

        verbose=True,

        calib_steps=len(dataset),

        collate_fn=collate_fn,

        executor=executor
    )


    print(
        "\nQuantization finished."
    )


    print(
        f"Operations after quantization: "
        f"{len(graph.operations)}"
    )


    # ========================================================
    # Graphwise Error
    # ========================================================

    graphwise_analysis(
        graph,
        dataset
    )


    # ========================================================
    # Layerwise Error
    # ========================================================

    layerwise_analysis(
        graph,
        dataset
    )


    # ========================================================
    # Output paths
    # ========================================================

    onnx_out = os.path.join(

        OUTPUT_DIR,

        "yolov8n_int8_ppq.onnx"
    )

    config_out = os.path.join(

        OUTPUT_DIR,

        "yolov8n_int8_ppq.json"
    )


    # ========================================================
    # Export
    # ========================================================

    print(
        "\nExporting PPQ TensorRT INT8 graph..."
    )

    export_ppq_graph(

        graph=graph,

        platform=TargetPlatform.TRT_INT8,

        graph_save_to=onnx_out,

        config_save_to=config_out,
    )


    print(
        "\n========== Done =========="
    )

    print(
        "INT8 ONNX:",
        onnx_out
    )

    print(
        "Config   :",
        config_out
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()