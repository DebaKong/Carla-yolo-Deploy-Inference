# 基于 CARLA + YOLOv8 + TensorRT 的实时目标检测与多目标跟踪系统

> Windows 11（CARLA 仿真 + 客户端）与 WSL2 Ubuntu 22.04（TensorRT 推理服务端）联合部署
> 端到端链路：**CARLA 相机采图 → TCP 传输 JPEG → TensorRT 推理 → ByteTrack 跟踪 → 回传结果并可视化**

---

## 一、这个项目能做什么

1. **仿真环境实时感知**：在 CARLA 仿真器中生成自车、车辆与行人，采集前视 RGB 相机画面（默认 1280×720），实时做目标检测。
2. **多精度 TensorRT 推理**：同一份 `yolov8n.onnx`，一键构建 **FP32 / FP16 / INT8(校准量化) / INT8(PPQ 量化)** 四种 engine，用于在精度与速度之间做取舍：
   - FP32：精度基准，体积大、延迟最高；
   - FP16：体积约减半、速度明显提升，精度损失通常很小；
   - INT8（TensorRT 校准量化）：显存占用与延迟进一步下降，需要用校准集标定激活值范围；
   - INT8（PPQ 量化）：由 PPQ 离线算出量化参数并导出到 ONNX，再用 TensorRT 构建，通常比朴素校准量化更可控。
3. **精度量化打平（mAP 评测）**：每构建完一个 engine 立刻在同一套数据集上跑一次评测，输出 **mAP@0.50 / mAP@0.50:0.95**，最后给出横向对比汇总表。
4. **多目标跟踪**：推理结果接入 ByteTrack（按类别分别跟踪），输出带 ID 的稳定轨迹，跨帧目标不会跳 ID。
5. **Windows ⇄ WSL 跨系统联合运行**：CARLA 跑在 Windows（图形/GPU 渲染），推理服务跑在 WSL Ubuntu（CUDA + TensorRT），两者通过 TCP 通信，Windows 端 OpenCV 窗口实时显示带框的结果。
6. **INT8 校准数据集自动生成**：提供专门的数据集采集脚本，在 CARLA 中自动布场景、自动生成车流人流，直接产出 YOLO 格式的图片 + 标签，可直接用于 INT8 校准和 mAP 评测。

---

## 二、系统架构

```
┌──────────────────────── Windows 11 ────────────────────────┐
│                                                            │
│   CARLA Simulator (0.9.x)                                  │
│        │  CARLA PythonAPI (127.0.0.1:2000)                 │
│        ▼                                                   │
│   carla_client/ (Python 复制副本)                          │
│     ├─ carla_world.py       连接仿真器 / 生成自车           │
│     ├─ carla_camera.py      RGB 相机 + JPEG 编码            │
│     ├─ carla_visualization.py  画框显示               ▲    │
│     ├─ carla_tcp.py         TCP 客户端                 │    │
│     └─ carla_demo.py        主程序，OpenCV 窗口显示     │    │
│                    │                                  │    │
│                    │  JPEG 帧 →           检测结果 ←   │    │
└────────────────────┼──────────────────────────────────┼────┘
                     │   TCP 5000                       │
                     │   4 字节大端长度 + payload        │
┌────────────────────┼──────────────────────────────────┼────┐
│                    ▼                                  │    │
│   WSL2 Ubuntu 22.04                                   │    │
│   build/tcp_server  <───┐                              │    │
│     ├─ yolov8.cpp       │  TensorRT 推理（加载 .engine）│    │
│     ├─ tracker.cpp      │  ByteTrack 多类别跟踪        │    │
│     └─ protocol.cpp     │  结果序列化 ─────────────────┘    │
│                                                            │
│   GPU: CUDA / TensorRT（含 trtexec）                       │
└────────────────────────────────────────────────────────────┘
```

**通信协议（自定义，很轻量）**

- 上行：客户端 → 服务端，`!I`（4 字节大端长度）+ JPEG 图片字节流。
- 下行：服务端 → 客户端，`!I`（4 字节大端长度）+ payload；payload 格式为 `count(int32 LE)` + 每个目标 28 字节，依次为 `class_id(int32) / track_id(int32) / confidence(float32) / x(int32) / y(int32) / width(int32) / height(int32)`。

---

## 三、目录结构

```
self_driving_project/
├── CMakeLists.txt                  # 构建 4 个 C++ 目标
├── include/                        # yolov8.h / tracker.h / protocol.h / tcp_server.h / Int8Calibrator.h / PPQInt8EngineBuilder.h
├── src/                            # 各目标源码
├── third_party/ByteTrack           # ByteTrack C++ 实现
├── assets/
│   ├── yolov8n.onnx                # FP32 ONNX（构建 engine 的输入）
│   ├── Calibration/                # 校准集 + 评测集（Windows 端生成后拷过来）
│   │   ├── images/ labels/ metadata.csv
│   │   └── calibration.cache       # INT8 校准缓存
│   └── *.engine                    # 构建产物
├── carla_client/                   # Python 端（Windows 副本 + 量化工具）
│   ├── carla_demo.py               # 演示主程序（连接 WSL 推理服务）
│   ├── carla_world.py / carla_camera.py / carla_visualization.py / carla_tcp.py
│   ├── carla_int8_dataset_collector.py   # INT8 校准集采集
│   ├── ppq_quantization.py         # PPQ INT8 量化
│   ├── requirements.txt            # Python 依赖
│   └── ppq_output/                 # 量化产物
├── scripts/build_and_eval_all.sh   # 一键构建 + 评测
└── build/                          # cmake 构建目录
```

**C++ 构建产物**

| 可执行文件 | 作用 |
| --- | --- |
| `tcp_server` | 推理服务端：`./tcp_server <engine路径>`，监听 `0.0.0.0:5000` |
| `build_int8_engine` | INT8 校准量化构建（路径在程序内写死，须在项目根目录运行） |
| `ppq_build_int8_engine` | 依据 PPQ 导出的 ONNX/JSON 构建 INT8 engine |
| `evaluate_map` | mAP 评测：`./evaluate_map <engine路径>` |

---

## 四、环境准备

### 4.1 WSL2 Ubuntu 22.04 侧（推理端）

需要：**NVIDIA 驱动（Windows 主机安装即可，WSL 自动可用）**、**CUDA Toolkit 12.8**、**TensorRT 10.8**（需包含 `trtexec`）、OpenCV 4、Eigen3、Boost、CMake ≥ 3.18、GCC ≥ 9（C++17）。

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libopencv-dev libeigen3-dev libboost-all-dev
# CUDA Toolkit 与 TensorRT 请按 NVIDIA 官方文档安装到 /usr/local/cuda 与系统库路径

# 在 WSL 中确认 GPU 可用
nvidia-smi
trtexec --help   # 或 /usr/local/cuda/bin/trtexec --help
```

编译 C++ 部分（在项目根目录执行）：

```bash
cd ~/self_driving_project      # 换成你的实际路径
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
ls build/tcp_server build/evaluate_map build/build_int8_engine build/ppq_build_int8_engine
```

> CMakeLists 中默认头文件/库路径为 `/usr/local/cuda/include`、`/usr/local/cuda/lib64`。若你的 CUDA/TensorRT 装在别处，需要相应调整环境变量或库的搜索路径。

### 4.2 Windows 11 侧（仿真 + 客户端）

- [CARLA 模拟器](https://github.com/carla-simulator/carla)（推荐 0.9.15 / 0.9.16 发行版，Windows 包解压即用）
- Miniconda / Anaconda

后面的步骤会创建 Python ≥ 3.10 的虚拟环境并安装 `carla_client/requirements.txt`。

> CARLA PythonAPI 的版本必须与模拟器的版本一致，否则会出现协议不匹配或 `ImportError`。PyPI 上的 `carla` 客户端 wheel 提供 Python 3.10 / 3.11 / 3.12；使用老版本 CARLA 时请改用发行包自带的 `PythonAPI/carla/dist/*.whl`（详见 `requirements.txt` 中的注释）。

---

## 五、使用方法（完整五步）

### 第一步：在 Windows 端准备 Python 环境

把整个 `carla_client` 文件夹复制一份到 Windows（例如 `D:\carla_client`），放在 Windows 侧运行：

```powershell
# 1) 创建虚拟环境（Python >= 3.10，推荐 3.10/3.11）
conda create -n carla python=3.10 -y
conda activate carla

# 2) 安装依赖
cd D:\carla_client
pip install -r requirements.txt
```

依赖包含 `carla`（仿真器 API）、`opencv-python`、`numpy`，以及量化脚本需要的 `ppq`、`torch`、`onnx`、`protobuf`、`tqdm`。

> `ppq==0.6.6` 是官方 2023 年的最后一个 PyPI 版本，在 Python 3.12 上可能出现兼容性问题。若 `import ppq` 报错，建议单独建一个 **Python 3.8~3.10** 的环境只用于运行 `ppq_quantization.py`。

### 第二步：生成校准集并完成 PPQ 量化（CARLA 需保持运行）

先启动 CARLA（Windows）：

```powershell
D:\CARLA_0.9.16\WindowsNoEditor\CarlaUE4.exe      # 路径按实际安装位置调整
```

然后在这个 conda 环境中执行：

```powershell
conda activate carla
cd D:\carla_client

# 2.1 采集 INT8 校准集（脚本内部会自动生成车流/人流、自动布场景）
#    产物：Calibration\images\*.jpg、Calibration\labels\*.txt、Calibration\metadata.csv
python carla_int8_dataset_collector.py

# 2.2 执行 PPQ 量化（读取 ./Calibration/images 做校准）
#    产物：ppq_output\yolov8n_int8_ppq.onnx、ppq_output\yolov8n_int8_ppq.json
python ppq_quantization.py
```

> **顺序提示**：`ppq_quantization.py` 依赖 2.1 产出的 `Calibration/images`，所以先采集、再量化。若已有现成的校准集，也可直接跳到 2.2。

把产物拷贝到 WSL 的 `assets` 下（在 WSL 终端执行，路径按实际调整）：

```bash
cp -r /mnt/d/carla_client/Calibration            ~/self_driving_project/assets/
cp    /mnt/d/carla_client/ppq_output/yolov8n_int8_ppq.onnx ~/self_driving_project/assets/
cp    /mnt/d/carla_client/ppq_output/yolov8n_int8_ppq.json ~/self_driving_project/assets/

# 校验
ls ~/self_driving_project/assets/Calibration            # images / labels / metadata.csv
ls ~/self_driving_project/assets | grep ppq
```

### 第三步：在 WSL 端构建 4 个 engine 并评测

在项目根目录执行（脚本会先切到项目根目录，相对路径不用担心）：

```bash
cd ~/self_driving_project
chmod +x scripts/build_and_eval_all.sh
./scripts/build_and_eval_all.sh
```

脚本依次完成：**构建 → 校验 engine 非空 → 评测**，其中

| 精度 | 构建方式 | 输出 |
| --- | --- | --- |
| FP32 | `trtexec --onnx=assets/yolov8n.onnx --saveEngine=...` | `assets/yolov8n_fp32.engine` |
| FP16 | `trtexec --onnx=... --fp16 --saveEngine=...` | `assets/yolov8n_fp16.engine` |
| INT8 | `build/build_int8_engine`（TRT 校准量化） | `assets/yolov8n_int8.engine` |
| PPQ INT8 | `build/ppq_build_int8_engine <onnx> <json> <engine>` | `assets/yolov8n_int8_ppq.engine` |

每步评测的日志在 `scripts/logs/`（`<case>_build.log`、`<case>_eval.log`），结尾会打印汇总表，例如：

```
CASE       BUILD            EVAL         mAP@0.50     mAP@0.50:0.95
fp32       ok(31s)          ok(12s)      0.xxxx       0.xxxx
fp16       ok(28s)          ok(9s)       0.xxxx       0.xxxx
int8       ok(52s)          ok(8s)       0.xxxx       0.xxxx
ppq_int8   ok(49s)          ok(8s)       0.xxxx       0.xxxx
```

常用参数：

```bash
./scripts/build_and_eval_all.sh --only=fp16,int8   # 只跑指定精度
./scripts/build_and_eval_all.sh --reuse            # engine 已存在则跳过构建
./scripts/build_and_eval_all.sh --eval-only        # 只评测，不重新构建
./scripts/build_and_eval_all.sh --no-eval          # 只构建，不评测
./scripts/build_and_eval_all.sh --no-bench         # trtexec 加 --buildOnly，不做推理测速
./scripts/build_and_eval_all.sh --help
```

环境变量可覆盖 `BIN_DIR`（可执行文件目录，默认自动探测 `./` 与 `./build/`）、`TRTEXEC`、`WORKSPACE_MB`、`ONNX`、`PPQ_ONNX`、`PPQ_JSON`、`OUT_DIR`。

> 注意：脚本会覆盖 `assets` 下已存在的同名 engine，想保留旧文件请加 `--reuse`。

### 第四步：在 WSL 端启动推理服务（CARLA 保持开启）

```bash
cd ~/self_driving_project
./build/tcp_server assets/yolov8n_int8_ppq.engine
# 也可以换成 assets/yolov8n_fp16.engine 等其它精度，逐个对比效果
```

看到 `[TCP] Waiting for client...` 即表示服务已就绪，等待 Windows 端连接。

同时查一下 WSL 的 IP，后面 Windows 端要填：

```bash
hostname -I          # 或：ip addr show eth0 | grep inet
```

### 第五步：Windows 端生成交通并运行演示

1. **先创建车流行人**（CARLA 自带示例脚本，`PythonAPI/examples/generate_traffic.py`）：

   ```powershell
   cd D:\CARLA_0.9.16\WindowsNoEditor\PythonAPI\examples
   python generate_traffic.py -n 30 -w 20      # 30 辆车、20 个行人，实际参数用 -h 查看
   ```

2. **再运行客户端**：在 Windows 端的 `carla_client` 副本里，把 `carla_demo.py` 中的 `WSL_IP` 改成上一步查到的 WSL 地址（当前默认是 `172.18.192.102`），然后新开一个终端：

   ```powershell
   conda activate carla
   cd D:\carla_client
   python carla_demo.py
   ```

   程序会连接 CARLA（`127.0.0.1:2000`）生成自车与相机，连接 WSL 推理服务（`<WSL_IP>:5000`），随后弹出 OpenCV 窗口显示实时检测结果（类别、跟踪 ID、置信度、检测框）。**按 `q` 键退出**，退出时会自动销毁相机与车辆。

> 客户端会在 Windows 上显示结果；Spectator 视角会自动跟随相机。若窗口但无画面，先看服务端日志是否有 `[Inference]` 输出。

---

## 六、常见问题排查

| 现象 | 可能原因 / 处理 |
| --- | --- |
| Windows 端 `ConnectionRefused` / 连不上 5000 | WSL IP 变了（WSL 重启后 IP 会变），重新 `hostname -I` 并修改 `carla_demo.py` 的 `WSL_IP`；或直接用 `localhost` 尝试（WSL 默认开启 localhost 转发） |
| `tcp_server` 报 `Usage: ./tcp_server <engine_path>` | 漏了 engine 路径参数 |
| engine 加载失败 | engine 与当前 GPU/驱动/TensorRT 版本强绑定，换机器或升级环境后需要重新构建 |
| `evaluate_map` 提示找不到图片 | `assets/Calibration/images` 没拷全，或没有以相对路径从项目根目录运行 |
| INT8 构建卡住/失败 | 检查 `assets/Calibration/images` 是否有图片；`build_int8_engine` 必须在项目根目录运行（路径写死） |
| `import ppq` 失败 | Python 版本过高，换 3.8~3.10 环境；检查 `protobuf<5` |
| CUDA / TensorRT 找不到 | 确认 `/usr/local/cuda` 存在、`nvidia-smi` 在 WSL 中可用 |
| 推理结果为 0 或延迟极高 | 确认选中的是 engine 而非 onnx；首次推理含 CUDA 初始化，属于正常抖动 |

---

## 七、备注与限制

- 可视化默认只标注 `person / car / motorcycle / bus / truck` 5 类（见 `carla_visualization.py` 的类别映射），其余 COCO 类别仍会参与推理与跟踪，只是不显示名称。
- `carla_demo.py` 中 CARLA 地址（127.0.0.1:2000）、相机分辨率（1280×720、FOV 90、JPEG 质量 85）、WSL 地址端口（5000）均为文件内常量，改场景请直接改对应常量。
- `assets/calibration.cache` 是 INT8 校准缓存，删除后重新构建会重新校准（更慢但结果更“干净”）。
- FP32 ONNX 的来源不在本项目范围内（可用官方 ultralytics 导出 `yolov8n.onnx`），`requirements.txt` 未包含相关依赖。
