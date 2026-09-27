#!/usr/bin/env bash
#
# ============================================================
# build_and_eval_all.sh
#
# 用 assets/yolov8n.onnx 依次构建 4 种精度的 TensorRT engine，
# 每构建完一个就立刻用 evaluate_map 评测一次 mAP：
#
#   1) fp32      : trtexec (默认精度)
#   2) fp16      : trtexec --fp16
#   3) int8      : ./build_int8_engine          (TensorRT 校准量化)
#   4) ppq_int8  : ./ppq_build_int8_engine      (PPQ 量化)
#
# 注意：build_int8_engine / evaluate_map 内部使用相对路径
#       (assets/yolov8n.onnx、assets/Calibration/...)，
#       因此本脚本会先 cd 到项目根目录再执行。
#
# 用法：
#   ./scripts/build_and_eval_all.sh
#   ./scripts/build_and_eval_all.sh --only=fp16,int8
#   ./scripts/build_and_eval_all.sh --eval-only      # 不重新构建，只评测已有 engine
#   ./scripts/build_and_eval_all.sh --no-eval        # 只构建，不评测
#   ./scripts/build_and_eval_all.sh --reuse          # engine 已存在则跳过构建
#   ./scripts/build_and_eval_all.sh --help
#
# 可用环境变量覆盖：
#   BIN_DIR=build            可执行文件所在目录（默认自动探测 ./ 或 ./build/）
#   TRTEXEC=/usr/local/cuda/bin/trtexec
#   WORKSPACE_MB=4096        trtexec workspace 大小(MB)，设为 0 则不传该参数
#   ONNX=assets/yolov8n.onnx
#   PPQ_ONNX=assets/yolov8n_int8_ppq.onnx
#   PPQ_JSON=assets/yolov8n_int8_ppq.json
#   OUT_DIR=assets           fp32 / fp16 engine 输出目录
# ============================================================

set -u
set -o pipefail

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || {
    echo "[FATAL] 无法进入项目根目录: ${PROJECT_ROOT}"
    exit 1
}

# ------------------------------------------------------------
# 默认配置
# ------------------------------------------------------------

ONNX="${ONNX:-assets/yolov8n.onnx}"
PPQ_ONNX="${PPQ_ONNX:-assets/yolov8n_int8_ppq.onnx}"
PPQ_JSON="${PPQ_JSON:-assets/yolov8n_int8_ppq.json}"
OUT_DIR="${OUT_DIR:-assets}"
BIN_DIR="${BIN_DIR:-}"

WORKSPACE_MB="${WORKSPACE_MB:-4096}"

LOG_DIR="${SCRIPT_DIR}/logs"

ENGINE_FP32="${OUT_DIR}/yolov8n_fp32.engine"
ENGINE_FP16="${OUT_DIR}/yolov8n_fp16.engine"
ENGINE_INT8="${OUT_DIR}/yolov8n_int8.engine"        # build_int8_engine 内部写死
ENGINE_PPQ="${OUT_DIR}/yolov8n_int8_ppq.engine"

ONLY=""
DO_BUILD=1
DO_EVAL=1
REUSE=0
NO_BENCH=0

# ------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------

usage() {
    cat <<'USAGE'
build_and_eval_all.sh — 依次构建 fp32 / fp16 / int8 / ppq_int8 engine 并逐个评测 mAP

用法：
  ./scripts/build_and_eval_all.sh                  全部构建 + 全部评测
  ./scripts/build_and_eval_all.sh --only=fp16,int8 只跑指定精度 (fp32,fp16,int8,ppq_int8)
  ./scripts/build_and_eval_all.sh --eval-only      不构建，只评测已有 engine
  ./scripts/build_and_eval_all.sh --no-eval        只构建，不评测
  ./scripts/build_and_eval_all.sh --reuse          engine 已存在则跳过构建
  ./scripts/build_and_eval_all.sh --no-bench       trtexec 加 --buildOnly，不做推理测速

环境变量覆盖：
  BIN_DIR=build             可执行文件目录 (默认自动探测 ./ 或 ./build/)
  TRTEXEC=/usr/local/cuda/bin/trtexec
  WORKSPACE_MB=4096         trtexec workspace(MB)，0 表示不传该参数
  ONNX=assets/yolov8n.onnx
  PPQ_ONNX=assets/yolov8n_int8_ppq.onnx
  PPQ_JSON=assets/yolov8n_int8_ppq.json
  OUT_DIR=assets            fp32 / fp16 engine 输出目录

注意：build_int8_engine 与 evaluate_map 内部使用相对路径，脚本会先 cd 到项目根目录。
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --only=*)
            ONLY="${1#*=}"
            ;;
        --eval-only)
            DO_BUILD=0
            ;;
        --no-eval)
            DO_EVAL=0
            ;;
        --reuse)
            REUSE=1
            ;;
        --no-bench)
            NO_BENCH=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

log_line() {
    echo "------------------------------------------------------------"
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

die() {
    echo ""
    echo "[FATAL] $*"
    echo ""
    exit 1
}

wanted() {
    local name="$1"
    [ -z "${ONLY}" ] && return 0
    case ",${ONLY}," in
        *",${name},"*) return 0 ;;
        *) return 1 ;;
    esac
}

# 探测可执行文件：优先 BIN_DIR，其次项目根目录，其次 build/
resolve_bin() {
    local name="$1"
    if [ -n "${BIN_DIR}" ] && [ -x "${BIN_DIR}/${name}" ]; then
        printf '%s\n' "${BIN_DIR}/${name}"
        return 0
    fi
    if [ -x "./${name}" ]; then
        printf '%s\n' "./${name}"
        return 0
    fi
    if [ -x "./build/${name}" ]; then
        printf '%s\n' "./build/${name}"
        return 0
    fi
    return 1
}

# 执行并把输出同时写入日志，返回命令自身退出码
run_logged() {
    local tag="$1"
    shift
    local logfile="${LOG_DIR}/${tag}.log"

    "$@" 2>&1 | tee "${logfile}"
    return "${PIPESTATUS[0]}"
}

extract_metric() {
    local logfile="$1"
    local pattern="$2"
    if [ -f "${logfile}" ]; then
        grep -m1 -E "${pattern}" "${logfile}" | awk -F': ' '{print $2}' | tr -d ' \r'
    fi
}

# ------------------------------------------------------------
# 结果汇总
# ------------------------------------------------------------

FAILED_COUNT=0

RESULT_NAME=()
RESULT_BUILD=()
RESULT_EVAL=()
RESULT_MAP50=()
RESULT_MAP95=()

record() {
    RESULT_NAME+=("$1")
    RESULT_BUILD+=("$2")
    RESULT_EVAL+=("$3")
    RESULT_MAP50+=("$4")
    RESULT_MAP95+=("$5")
}

# ------------------------------------------------------------
# 单个 case：构建 -> 校验 -> 评测
# ------------------------------------------------------------

run_case() {
    local name="$1"
    local engine="$2"
    shift 2

    local t0 t1 dur
    local build_status="-"
    local eval_status="-"
    local map50="-"
    local map95="-"

    log_line
    log "CASE: ${name}"
    log "engine: ${engine}"
    log_line

    wanted "${name}" || {
        log "跳过 (不在 --only 列表中)"
        return 0
    }

    # ---------- 构建 ----------

    if [ "${DO_BUILD}" -eq 1 ]; then
        if [ "${REUSE}" -eq 1 ] && [ -s "${engine}" ]; then
            log "engine 已存在且指定了 --reuse，跳过构建"
            build_status="reused"
        else
            t0=$(date +%s)
            if run_logged "${name}_build" "$@"; then
                t1=$(date +%s)
                dur=$((t1 - t0))
                if [ -s "${engine}" ]; then
                    build_status="ok(${dur}s)"
                    log "构建成功，用时 ${dur}s，大小: $(du -h "${engine}" | cut -f1)"
                else
                    build_status="FAILED(no output)"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                    log "构建命令返回 0，但未生成 engine 文件: ${engine}"
                fi
            else
                build_status="FAILED"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                log "构建失败，日志: ${LOG_DIR}/${name}_build.log"
            fi
        fi
    else
        build_status="skipped"
    fi

    # ---------- 校验 ----------

    if [ ! -s "${engine}" ]; then
        log "engine 不存在或为空，跳过评测"
        record "${name}" "${build_status}" "skipped" "-" "-" "-"
        return 0
    fi

    # ---------- 评测 ----------

    if [ "${DO_EVAL}" -eq 1 ]; then
        t0=$(date +%s)
        if run_logged "${name}_eval" "${EVAL_BIN}" "${engine}"; then
            t1=$(date +%s)
            dur=$((t1 - t0))
            eval_status="ok(${dur}s)"
            map50="$(extract_metric "${LOG_DIR}/${name}_eval.log" '^mAP@0\.50 +:')"
            map95="$(extract_metric "${LOG_DIR}/${name}_eval.log" '^mAP@0\.50:0\.95 +:')"
            [ -z "${map50}" ] && map50="-"
            [ -z "${map95}" ] && map95="-"
            log "评测完成: mAP@0.50=${map50}  mAP@0.50:0.95=${map95}"
        else
            eval_status="FAILED"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            log "评测失败，日志: ${LOG_DIR}/${name}_eval.log"
        fi
    else
        eval_status="skipped"
    fi

    record "${name}" "${build_status}" "${eval_status}" "${map50}" "${map95}"
}

# ------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------

mkdir -p "${LOG_DIR}"

log "项目根目录: ${PROJECT_ROOT}"
log "日志目录  : ${LOG_DIR}"

[ -f "${ONNX}" ] || die "找不到 ONNX 模型: ${ONNX}"
[ -f "${PPQ_ONNX}" ] || die "找不到 PPQ 量化 ONNX: ${PPQ_ONNX}"
[ -f "${PPQ_JSON}" ] || die "找不到 PPQ 量化信息: ${PPQ_JSON}"

EVAL_BIN="$(resolve_bin evaluate_map)" \
    || die "找不到 evaluate_map，请先编译 (cmake --build build) 或用 BIN_DIR= 指定目录"

if [ "${DO_BUILD}" -eq 1 ]; then
    INT8_BIN="$(resolve_bin build_int8_engine)" \
        || die "找不到 build_int8_engine，请先编译或用 BIN_DIR= 指定目录"
    PPQ_BIN="$(resolve_bin ppq_build_int8_engine)" \
        || die "找不到 ppq_build_int8_engine，请先编译或用 BIN_DIR= 指定目录"
else
    INT8_BIN="build_int8_engine"
    PPQ_BIN="ppq_build_int8_engine"
fi

log "evaluate_map          : ${EVAL_BIN}"
log "build_int8_engine     : ${INT8_BIN}"
log "ppq_build_int8_engine : ${PPQ_BIN}"

# ---------- trtexec ----------

TRTEXEC_WS=""

if [ "${DO_BUILD}" -eq 1 ] && { wanted fp32 || wanted fp16; }; then
    if [ -z "${TRTEXEC:-}" ]; then
        if command -v trtexec >/dev/null 2>&1; then
            TRTEXEC="$(command -v trtexec)"
        elif [ -x /usr/local/cuda/bin/trtexec ]; then
            TRTEXEC=/usr/local/cuda/bin/trtexec
        elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
            TRTEXEC=/usr/src/tensorrt/bin/trtexec
        fi
    fi

    [ -n "${TRTEXEC:-}" ] && [ -x "${TRTEXEC}" ] \
        || die "找不到 trtexec，请用 TRTEXEC=/path/to/trtexec 指定"

    log "trtexec               : ${TRTEXEC}"

    if [ "${WORKSPACE_MB}" -gt 0 ]; then
        # TensorRT 10 用 --memPoolSize，老版本用 --workspace
        if "${TRTEXEC}" --help 2>&1 | grep -q -- "--memPoolSize"; then
            TRTEXEC_WS="--memPoolSize=workspace:$((WORKSPACE_MB * 1024 * 1024))"
        elif "${TRTEXEC}" --help 2>&1 | grep -q -- "--workspace"; then
            TRTEXEC_WS="--workspace=${WORKSPACE_MB}"
        fi
    fi
fi

[ "${NO_BENCH}" -eq 1 ] && TRTEXEC_BENCH_FLAG="--buildOnly" || TRTEXEC_BENCH_FLAG=""

# ------------------------------------------------------------
# 开始执行
# ------------------------------------------------------------

START_TIME=$(date +%s)

# 1) FP32
run_case "fp32" "${ENGINE_FP32}" \
    "${TRTEXEC:-true}" --onnx="${ONNX}" --saveEngine="${ENGINE_FP32}" \
    ${TRTEXEC_WS} ${TRTEXEC_BENCH_FLAG}

# 2) FP16
run_case "fp16" "${ENGINE_FP16}" \
    "${TRTEXEC:-true}" --onnx="${ONNX}" --fp16 --saveEngine="${ENGINE_FP16}" \
    ${TRTEXEC_WS} ${TRTEXEC_BENCH_FLAG}

# 3) INT8 (TensorRT 校准量化，路径在程序内写死)
run_case "int8" "${ENGINE_INT8}" \
    "${INT8_BIN}"

# 4) PPQ INT8
run_case "ppq_int8" "${ENGINE_PPQ}" \
    "${PPQ_BIN}" "${PPQ_ONNX}" "${PPQ_JSON}" "${ENGINE_PPQ}"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# ------------------------------------------------------------
# 汇总
# ------------------------------------------------------------

log_line
echo ""
echo "==================== 汇总 ===================="
printf "%-10s %-16s %-12s %-12s %-12s\n" \
    "CASE" "BUILD" "EVAL" "mAP@0.50" "mAP@0.50:0.95"

i=0
while [ $i -lt ${#RESULT_NAME[@]} ]; do
    printf "%-10s %-16s %-12s %-12s %-12s\n" \
        "${RESULT_NAME[$i]}" "${RESULT_BUILD[$i]}" "${RESULT_EVAL[$i]}" \
        "${RESULT_MAP50[$i]}" "${RESULT_MAP95[$i]}"
    i=$((i + 1))
done

echo ""
echo "总耗时: ${TOTAL_TIME}s"
echo "日志目录: ${LOG_DIR}"
echo "=============================================="

if [ "${FAILED_COUNT}" -gt 0 ]; then
    echo ""
    echo "[WARN] 有 ${FAILED_COUNT} 个阶段失败，请查看 ${LOG_DIR} 下的日志。"
    exit 1
fi

exit 0
