# Real-Time Object Detection and Multi-Object Tracking with CARLA + YOLOv8 + TensorRT

> Distributed deployment across **Windows 11** (CARLA simulation + client) and **WSL2 Ubuntu 22.04** (TensorRT inference server)
> End-to-end pipeline: **CARLA camera capture → JPEG over TCP → TensorRT inference → ByteTrack tracking → results returned and visualized**

[简体中文](./README.zh-CN.md) | English

---

## 1. What This Project Does

1. **Real-time perception in simulation** — spawns an ego vehicle, surrounding vehicles and pedestrians in CARLA, captures the front RGB camera stream (1280×720 by default) and runs object detection on it in real time.
2. **Multi-precision TensorRT inference** — from a single `yolov8n.onnx`, builds four engines in one shot: **FP32 / FP16 / INT8 (calibration) / INT8 (PPQ)**, so you can trade off accuracy against speed:
   - **FP32** — accuracy baseline; largest model, highest latency.
   - **FP16** — roughly half the model size with a clear speed-up; usually negligible accuracy loss.
   - **INT8 (TensorRT calibration)** — further reduces memory footprint and latency; requires a calibration set to determine activation ranges.
   - **INT8 (PPQ)** — quantization parameters are computed offline by PPQ and exported into the ONNX graph, then built with TensorRT; generally more controllable than plain calibration-based quantization.
3. **Accuracy benchmarking (mAP)** — every engine is evaluated on the same dataset right after it is built, reporting **mAP@0.50 / mAP@0.50:0.95**, followed by a side-by-side summary table.
4. **Multi-object tracking** — detections are fed into ByteTrack (one tracker per class), producing stable trajectories with persistent IDs across frames.
5. **Cross-OS pipeline: Windows ⇄ WSL** — CARLA runs on Windows (rendering), the inference server runs on WSL Ubuntu (CUDA + TensorRT); the two communicate over TCP and Windows displays the annotated result live in an OpenCV window.
6. **Automatic INT8 calibration dataset generation** — a dedicated collector script sets up scenes and spawns traffic/pedestrians inside CARLA, and directly outputs YOLO-format images + labels usable for both INT8 calibration and mAP evaluation.

---

## 2. Architecture

```
+------------------------------- Windows 11 --------------------------------+
|                                                                           |
|   CARLA Simulator (0.9.x)                                                 |
|        |   CARLA PythonAPI (127.0.0.1:2000)                               |
|        v                                                                  |
|   carla_client/  (Python copy)                                            |
|     +- carla_world.py          connect to simulator / spawn ego vehicle   |
|     +- carla_camera.py         RGB camera + JPEG encoding            ^    |
|     +- carla_visualization.py  draw boxes & show window              |    |
|     +- carla_tcp.py            TCP client                            |    |
|     +- carla_demo.py           main app, OpenCV window               |    |
|                    |                                                 |    |
|                    |  JPEG frames ->          detections  <----------+    |
+--------------------|-------------------------------------------------+----+
                     |   TCP port 5000
                     |   4-byte big-endian length + payload
+--------------------|-------------------------------------------------+----+
|                    v                                                 |    |
|   WSL2 Ubuntu 22.04                                                  |    |
|   build/tcp_server  <----+                                           |    |
|     +- yolov8.cpp        |  TensorRT inference (loads .engine)        |    |
|     +- tracker.cpp       |  ByteTrack per-class tracking             |    |
|     +- protocol.cpp      |  serialize results -----------------------+    |
|                                                                           |
|   GPU: CUDA 12.8 / TensorRT 10.8 (including trtexec)                      |
+---------------------------------------------------------------------------+
```

**Wire protocol (custom and very lightweight)**

- Uplink (client → server): `!I` (4-byte big-endian length) + JPEG bytes.
- Downlink (server → client): `!I` (4-byte big-endian length) + payload, where payload = `count (int32 LE)` followed by 28 bytes per object: `class_id (int32) / track_id (int32) / confidence (float32) / x (int32) / y (int32) / width (int32) / height (int32)`.

---

## 3. Repository Layout

```
self_driving_project/
├── CMakeLists.txt                  # builds 4 C++ targets
├── include/                        # yolov8.h / tracker.h / protocol.h / tcp_server.h / Int8Calibrator.h / PPQInt8EngineBuilder.h
├── src/                            # sources of all targets
├── third_party/ByteTrack           # ByteTrack C++ implementation
├── assets/
│   ├── yolov8n.onnx                # FP32 ONNX (input for engine building)
│   ├── Calibration/                # calibration + evaluation set (copied from Windows)
│   │   ├── images/ labels/ metadata.csv
│   │   └── calibration.cache       # INT8 calibration cache
│   └── *.engine                    # build artifacts
├── carla_client/                   # Python side (Windows copy + quantization tools)
│   ├── carla_demo.py               # demo entry point (connects to the WSL inference server)
│   ├── carla_world.py / carla_camera.py / carla_visualization.py / carla_tcp.py
│   ├── carla_int8_dataset_collector.py   # INT8 calibration set collector
│   ├── ppq_quantization.py         # PPQ INT8 quantization
│   ├── requirements.txt            # Python dependencies
│   └── ppq_output/                 # quantization artifacts
├── scripts/build_and_eval_all.sh   # one-shot build + evaluate
└── build/                          # cmake build directory
```

**C++ build artifacts**

| Executable | Purpose |
| --- | --- |
| `tcp_server` | Inference server: `./tcp_server <engine_path>`, listens on `0.0.0.0:5000` |
| `build_int8_engine` | INT8 calibration-based build (paths are hard-coded, must run from the project root) |
| `ppq_build_int8_engine` | Builds an INT8 engine from the PPQ-exported ONNX/JSON |
| `evaluate_map` | mAP evaluation: `./evaluate_map <engine_path>` |

---

## 4. Environment Setup

### 4.1 WSL2 Ubuntu 22.04 (inference side)

Requirements: **NVIDIA driver** (installed on the Windows host, automatically usable inside WSL), **CUDA Toolkit 12.8**, **TensorRT 10.8** (must include `trtexec`), OpenCV 4, Eigen3, Boost, CMake ≥ 3.18, GCC ≥ 9 (C++17).

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libopencv-dev libeigen3-dev libboost-all-dev
# Install CUDA Toolkit and TensorRT following the official NVIDIA documentation
# (CUDA under /usr/local/cuda, TensorRT into the system library path)

# Verify the GPU is visible inside WSL
nvidia-smi
trtexec --help   # or: /usr/local/cuda/bin/trtexec --help
```

Build the C++ part (from the project root):

```bash
cd ~/self_driving_project      # adjust to your actual path
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
ls build/tcp_server build/evaluate_map build/build_int8_engine build/ppq_build_int8_engine
```

> `CMakeLists.txt` expects headers/libraries under `/usr/local/cuda/include` and `/usr/local/cuda/lib64`. If CUDA/TensorRT is installed elsewhere, adjust the environment variables or the library search path accordingly.

### 4.2 Windows 11 (simulation + client side)

- [CARLA simulator](https://github.com/carla-simulator/carla) (release builds such as 0.9.15 / 0.9.16 are recommended; the Windows package is ready to run after extracting)
- Miniconda / Anaconda

The next steps create a Python ≥ 3.10 virtual environment and install `carla_client/requirements.txt`.

> The CARLA PythonAPI version must match the simulator version, otherwise you will hit protocol mismatches or an `ImportError`. The `carla` client wheel on PyPI targets Python 3.10 / 3.11 / 3.12; for older CARLA releases use the bundled `PythonAPI/carla/dist/*.whl` instead (see the notes in `requirements.txt`).

---

## 5. Usage (Five Steps)

### Step 1 — Prepare the Python environment on Windows

Copy the whole `carla_client` folder to the Windows side (e.g. `D:\carla_client`) and run:

```powershell
# 1) Create the virtual environment (Python >= 3.10; 3.10 / 3.11 recommended)
conda create -n carla python=3.10 -y
conda activate carla

# 2) Install dependencies
cd D:\carla_client
pip install -r requirements.txt
```

Dependencies include `carla` (simulator API), `opencv-python` and `numpy`, plus `ppq`, `torch`, `onnx`, `protobuf` and `tqdm` for the quantization script.

> `ppq==0.6.6` is the last release published to PyPI (2023) and may have compatibility issues on Python 3.12. If `import ppq` fails, create a separate **Python 3.8–3.10** environment dedicated to running `ppq_quantization.py`.

### Step 2 — Generate the calibration set and run PPQ quantization (keep CARLA running)

Start CARLA first (Windows):

```powershell
D:\CARLA_0.9.16\WindowsNoEditor\CarlaUE4.exe      # adjust to your installation path
```

Then, in the same conda environment:

```powershell
conda activate carla
cd D:\carla_client

# 2.1 Collect the INT8 calibration set (the script spawns traffic/pedestrians and scenes itself)
#     Output: Calibration\images\*.jpg, Calibration\labels\*.txt, Calibration\metadata.csv
python carla_int8_dataset_collector.py

# 2.2 Run PPQ quantization (reads ./Calibration/images for calibration)
#     Output: ppq_output\yolov8n_int8_ppq.onnx, ppq_output\yolov8n_int8_ppq.json
python ppq_quantization.py
```

> **Order matters**: `ppq_quantization.py` depends on the `Calibration/images` produced in step 2.1, so collect first and quantize afterwards. If you already have a calibration set, you can jump directly to 2.2.
> Note that `ppq_quantization.py` expects the FP32 model at `./ppq_output/yolov8n.onnx` (`ONNX_PATH` at the top of the file) — place it there beforehand if it is missing.

Copy the artifacts into `assets/` on the WSL side (run in a WSL terminal, adjust paths):

```bash
cp -r /mnt/d/carla_client/Calibration            ~/self_driving_project/assets/
cp    /mnt/d/carla_client/ppq_output/yolov8n_int8_ppq.onnx ~/self_driving_project/assets/
cp    /mnt/d/carla_client/ppq_output/yolov8n_int8_ppq.json ~/self_driving_project/assets/

# Verify
ls ~/self_driving_project/assets/Calibration            # images / labels / metadata.csv
ls ~/self_driving_project/assets | grep ppq
```

### Step 3 — Build the four engines and evaluate them (WSL side)

Run from the project root (the script `cd`s to the project root itself, so relative paths are safe):

```bash
cd ~/self_driving_project
chmod +x scripts/build_and_eval_all.sh
./scripts/build_and_eval_all.sh
```

For each precision the script performs: **build → verify the engine is non-empty → evaluate**.

| Precision | How it is built | Output |
| --- | --- | --- |
| FP32 | `trtexec --onnx=assets/yolov8n.onnx --saveEngine=...` | `assets/yolov8n_fp32.engine` |
| FP16 | `trtexec --onnx=... --fp16 --saveEngine=...` | `assets/yolov8n_fp16.engine` |
| INT8 | `build/build_int8_engine` (TRT calibration) | `assets/yolov8n_int8.engine` |
| PPQ INT8 | `build/ppq_build_int8_engine <onnx> <json> <engine>` | `assets/yolov8n_int8_ppq.engine` |

Logs for every stage go to `scripts/logs/` (`<case>_build.log`, `<case>_eval.log`), and a summary table is printed at the end, e.g.:

```
CASE       BUILD            EVAL         mAP@0.50     mAP@0.50:0.95
fp32       ok(31s)          ok(12s)      0.xxxx       0.xxxx
fp16       ok(28s)          ok(9s)       0.xxxx       0.xxxx
int8       ok(52s)          ok(8s)       0.xxxx       0.xxxx
ppq_int8   ok(49s)          ok(8s)       0.xxxx       0.xxxx
```

Common options:

```bash
./scripts/build_and_eval_all.sh --only=fp16,int8   # run only the given precisions
./scripts/build_and_eval_all.sh --reuse            # skip building if the engine already exists
./scripts/build_and_eval_all.sh --eval-only        # evaluate existing engines only, do not rebuild
./scripts/build_and_eval_all.sh --no-eval          # build only, skip evaluation
./scripts/build_and_eval_all.sh --no-bench         # pass --buildOnly to trtexec, skip latency benchmarking
./scripts/build_and_eval_all.sh --help
```

These can be overridden through environment variables: `BIN_DIR` (executable directory, auto-detected as `./` or `./build/`), `TRTEXEC`, `WORKSPACE_MB`, `ONNX`, `PPQ_ONNX`, `PPQ_JSON`, `OUT_DIR`.

> The script overwrites engines with the same name under `assets/`; add `--reuse` if you want to keep existing files.

### Step 4 — Start the inference server on WSL (keep CARLA running)

```bash
cd ~/self_driving_project
./build/tcp_server assets/yolov8n_int8_ppq.engine
# swap in assets/yolov8n_fp16.engine etc. to compare precisions
```

Once you see `[TCP] Waiting for client...`, the server is ready and waiting for the Windows client.

Also get the WSL IP address — it is needed on the Windows side:

```bash
hostname -I          # or: ip addr show eth0 | grep inet
```

### Step 5 — Spawn traffic on Windows and run the demo

1. **Create vehicles and pedestrians first**, using the example script shipped with CARLA (`PythonAPI/examples/generate_traffic.py`):

   ```powershell
   cd D:\CARLA_0.9.16\WindowsNoEditor\PythonAPI\examples
   python generate_traffic.py -n 30 -w 20      # 30 vehicles, 20 walkers; run with -h for all options
   ```

2. **Then run the client**: in the Windows copy of `carla_client`, set `WSL_IP` in `carla_demo.py` to the WSL address from the previous step (the current default is `172.18.192.102`), then open a new terminal:

   ```powershell
   conda activate carla
   cd D:\carla_client
   python carla_demo.py
   ```

   The app connects to CARLA (`127.0.0.1:2000`) to spawn the ego vehicle and camera, connects to the WSL inference server (`<WSL_IP>:5000`), and opens an OpenCV window showing live detections (class, track ID, confidence and bounding box). **Press `q` to quit**; the camera and vehicle are destroyed automatically on exit.

> Results are displayed on Windows and the spectator view follows the camera automatically. If the window appears but nothing shows up, check whether the server prints `[Inference]` lines.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `ConnectionRefused` on Windows / cannot reach port 5000 | The WSL IP changed (it does after every WSL restart) — re-run `hostname -I` and update `WSL_IP` in `carla_demo.py`; alternatively try `localhost` (WSL enables localhost forwarding by default) |
| `tcp_server` prints `Usage: ./tcp_server <engine_path>` | Missing the engine path argument |
| Engine fails to load | Engines are tied to the exact GPU / driver / TensorRT version; rebuild after changing machine or upgrading the stack |
| `evaluate_map` cannot find images | `assets/Calibration/images` was not copied completely, or the program was not run from the project root (relative paths) |
| INT8 build hangs or fails | Check that `assets/Calibration/images` is populated; `build_int8_engine` must be run from the project root (hard-coded paths) |
| `import ppq` fails | Python version too new — switch to a 3.8–3.10 environment; also check that `protobuf<5` |
| CUDA / TensorRT not found | Make sure `/usr/local/cuda` exists and `nvidia-smi` works inside WSL |
| No detections or very high latency | Confirm you passed an `.engine` file rather than an ONNX; the first inference includes CUDA initialization overhead |

---

## 7. Notes and Limitations

- Only five classes are labelled in the visualization by default (`person / car / motorcycle / bus / truck`, see the class map in `carla_visualization.py`); other COCO classes still go through inference and tracking, they just have no display name.
- In `carla_demo.py`, the CARLA address (127.0.0.1:2000), camera settings (1280×720, FOV 90, JPEG quality 85) and the WSL address/port (5000) are hard-coded constants — edit them directly to change the scenario.
- `assets/calibration.cache` is the INT8 calibration cache; deleting it forces recalibration on the next build (slower, but cleaner results).
- Producing the FP32 `yolov8n.onnx` is out of scope for this project (e.g. export it with the official Ultralytics package); `requirements.txt` does not include dependencies for that step.
