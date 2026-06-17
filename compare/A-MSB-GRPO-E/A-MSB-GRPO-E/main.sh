#!/usr/bin/env bash
# ==============================================================================
# main.sh — A-MSB-GRPO Full Pipeline Runner
# ==============================================================================
#
# Chạy toàn bộ pipeline A-MSB-GRPO:
#   1. Chuẩn bị dataset (download & convert to parquet)
#   2. Cài đặt dependencies
#   3. Huấn luyện mô hình



#
# Cách dùng:
#   chmod +x main.sh
#   ./main.sh                    # Chạy full pipeline (mặc định)
#   ./main.sh --skip-data        # Bỏ qua bước tải dataset (đã tải trước đó)
#   ./main.sh --debug            # Chế độ debug: 5 mẫu, không DeepSpeed
#   ./main.sh --dry-run          # Chỉ in lệnh, không chạy
#   ./main.sh --skip-data --skip-install # đã có data và library
# ==============================================================================

set -euo pipefail


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ──────────────────────────────────────────────────────────────────────────────
# Đường dẫn (Paths)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Dataset paths
DATA_PREPARE_SCRIPT="${REPO_ROOT}/data/prepare_requested_datasets.py"
DATASET_ROOT="${REPO_ROOT}/data/datasets/requested"
TRAIN_PARQUET="${DATASET_ROOT}/train/combined_train.parquet"
TEST_PARQUET="${DATASET_ROOT}/test/combined_test.parquet"

# Pipeline paths
PIPELINE_DIR="${SCRIPT_DIR}"
TRAIN_SCRIPT="${PIPELINE_DIR}/run_train.py"
DS_CONFIG="${PIPELINE_DIR}/configs/ds_config_zero3.json"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
OUTPUT_DIR="${PIPELINE_DIR}/checkpoints"
LOG_DIR="${PIPELINE_DIR}/logs"

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình mặc định (Default Configuration)
# ──────────────────────────────────────────────────────────────────────────────

# Model
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
USE_QLORA="--use-qlora"

# A-MSB-GRPO hyperparameters
N_ROLLOUTS=8          # Layer 1: số rollouts per prompt
M_REFLECTIONS=2       # Layer 2: số nhánh self-reflection per rollout
BATCH_SIZE_K=8       # Kích thước batch tĩnh (50/50: 4 correct + 4 error)
N_CLUSTERS=4          # Số cụm K-Means cho error profiling

# Loss function
KL_COEFF=0.01
CLIP_EPS=0.2
VIRTUAL_REWARD=-1.0
ENTROPY_SCALE_MIN=0.01

# Training
NUM_EPOCHS=2
LEARNING_RATE=1e-5
MAX_NEW_TOKENS=1024
MAX_SEQ_LEN=2048

# vLLM
VLLM_GPU_UTIL=0.30

# Data
MAX_SAMPLES=-1        # -1 = toàn bộ dataset
MAX_TEST_SAMPLES=-1

# ──────────────────────────────────────────────────────────────────────────────
# Flags
# ──────────────────────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_INSTALL=false
DEBUG_MODE=false
DRY_RUN=false
USE_DEEPSPEED=false
RESUME_TRAINING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)      SKIP_DATA=true; shift ;;
        --skip-install)   SKIP_INSTALL=true; shift ;;
        --debug)          DEBUG_MODE=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --no-deepspeed)   USE_DEEPSPEED=false; shift ;;
        --resume)         RESUME_TRAINING=true; shift ;;
        --epochs)         NUM_EPOCHS="$2"; shift 2 ;;
        --max-samples)    MAX_SAMPLES="$2"; shift 2 ;;
        --model)          MODEL_NAME="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-data       Bỏ qua bước tải dataset"
            echo "  --skip-install    Bỏ qua bước cài đặt dependencies"
            echo "  --debug           Chế độ debug (5 mẫu, không DeepSpeed)"
            echo "  --dry-run         Chỉ in lệnh, không thực thi"
            echo "  --no-deepspeed    Chạy không có DeepSpeed"
            echo "  --resume          Tiếp tục training từ checkpoint gần nhất"
            echo "  --epochs N        Số epoch (mặc định: 3)"
            echo "  --max-samples N   Giới hạn số mẫu training (mặc định: -1 = all)"
            echo "  --model NAME      HuggingFace model name"
            echo "  -h, --help        Hiển thị trợ giúp"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Debug mode overrides
if [[ "$DEBUG_MODE" == "true" ]]; then
    MAX_SAMPLES=5
    MAX_TEST_SAMPLES=3
    NUM_EPOCHS=1
    USE_DEEPSPEED=false
    echo "⚙️  DEBUG MODE: 5 train samples, 1 epoch, no DeepSpeed"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          A-MSB-GRPO Training Pipeline                      ║"
echo "║  Adaptive Multi-Layer Semantic-Balanced GRPO               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_info "Timestamp:    ${TIMESTAMP}"
log_info "Model:        ${MODEL_NAME}"
log_info "Epochs:       ${NUM_EPOCHS}"
log_info "N (rollouts): ${N_ROLLOUTS}  |  M (reflections): ${M_REFLECTIONS}  |  K (batch): ${BATCH_SIZE_K}"
log_info "DeepSpeed:    ${USE_DEEPSPEED}"
log_info "Pipeline dir: ${PIPELINE_DIR}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: Cài đặt Dependencies
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == "false" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "PHASE 1: Cài đặt dependencies..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "$REQUIREMENTS" ]]; then
        run_cmd pip install --no-cache-dir -r "$REQUIREMENTS"
        log_success "Dependencies installed."
    else
        log_error "requirements.txt not found at: ${REQUIREMENTS}"
        exit 1
    fi
    echo ""
else
    log_warn "Skipping dependency installation (--skip-install)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: Chuẩn bị Dataset
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_DATA" == "false" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "PHASE 2: Chuẩn bị datasets..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "$TRAIN_PARQUET" && -f "$TEST_PARQUET" ]]; then
        log_warn "Datasets đã tồn tại. Bỏ qua download."
        log_info "  Train: ${TRAIN_PARQUET}"
        log_info "  Test:  ${TEST_PARQUET}"
    else
        if [[ ! -f "$DATA_PREPARE_SCRIPT" ]]; then
            log_error "Dataset prepare script not found: ${DATA_PREPARE_SCRIPT}"
            exit 1
        fi

        log_info "Downloading & preparing datasets..."
        run_cmd python3 "$DATA_PREPARE_SCRIPT" \
            --output-root "data/datasets/requested" \
            --manifest-path "datasets/requested_datasets.manifest.yaml" \
            --lock-path "datasets/requested_datasets.lock.json"

        if [[ -f "$TRAIN_PARQUET" ]]; then
            log_success "Datasets prepared successfully."
        else
            log_error "Dataset preparation failed! Train parquet not found."
            exit 1
        fi
    fi
    echo ""
else
    log_warn "Skipping dataset preparation (--skip-data)"
    if [[ ! -f "$TRAIN_PARQUET" ]]; then
        log_error "Train parquet not found: ${TRAIN_PARQUET}"
        log_error "Run without --skip-data first to download datasets."
        exit 1
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: Huấn luyện (Training)
# ──────────────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "PHASE 3: Bắt đầu huấn luyện A-MSB-GRPO..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Tạo thư mục output
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# Build training command
TRAIN_ARGS=(
    --train-parquet "$TRAIN_PARQUET"
    --test-parquet "$TEST_PARQUET"
    --model-name "$MODEL_NAME"
    $USE_QLORA
    --n-rollouts "$N_ROLLOUTS"
    --m-reflections "$M_REFLECTIONS"
    --batch-size-k "$BATCH_SIZE_K"
    --n-clusters "$N_CLUSTERS"
    --kl-coeff "$KL_COEFF"
    --clip-eps "$CLIP_EPS"
    --virtual-reward "$VIRTUAL_REWARD"
    --entropy-scale-min "$ENTROPY_SCALE_MIN"
    --num-epochs "$NUM_EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-seq-len "$MAX_SEQ_LEN"
    --vllm-gpu-util "$VLLM_GPU_UTIL"
    --output-dir "$OUTPUT_DIR"
    --log-file "$LOG_FILE"
)

# Thêm giới hạn mẫu nếu có
if [[ "$MAX_SAMPLES" != "-1" ]]; then
    TRAIN_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$MAX_TEST_SAMPLES" != "-1" ]]; then
    TRAIN_ARGS+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi
if [[ "$RESUME_TRAINING" == "true" ]]; then
    TRAIN_ARGS+=(--resume)
    log_info "Resume mode: ON — sẽ tiếp tục từ checkpoint gần nhất"
fi

# Chọn launcher: DeepSpeed hoặc Python
if [[ "$USE_DEEPSPEED" == "true" ]]; then
    log_info "Launcher: DeepSpeed"
    log_info "Config:   ${DS_CONFIG}"

    FULL_CMD=(
        deepspeed "$TRAIN_SCRIPT"
        --deepspeed "$DS_CONFIG"
        "${TRAIN_ARGS[@]}"
    )
else
    log_info "Launcher: Python (standalone, no DeepSpeed)"

    FULL_CMD=(
        python3 "$TRAIN_SCRIPT"
        --no-deepspeed
        "${TRAIN_ARGS[@]}"
    )
fi

log_info "Log file: ${LOG_FILE}"
log_info "Command:"
echo "  ${FULL_CMD[*]}"
echo ""

# Chạy training — log vào file, terminal chỉ hiện tqdm (tqdm in ra stderr)
cd "$PIPELINE_DIR"
"${FULL_CMD[@]}" >> "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$TRAIN_EXIT_CODE" -eq 0 ]]; then
    log_success "Training hoàn tất!"
    log_info "Checkpoints: ${OUTPUT_DIR}"
    log_info "Log file:    ${LOG_FILE}"
    log_info "Metrics:     ${OUTPUT_DIR}/training_metrics.json"
else
    log_error "Training thất bại với exit code: ${TRAIN_EXIT_CODE}"
    log_error "Xem log tại: ${LOG_FILE}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$TRAIN_EXIT_CODE"
