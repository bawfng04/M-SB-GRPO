#!/usr/bin/env bash
# ==============================================================================
# test.sh — A-MSB-GRPO Benchmark Evaluation on Sample Datasets
# ==============================================================================
#
# Chạy benchmark evaluation trên các tập dataset sample trong:
#   data/datasets/test/benchmark-dryrun-amsb/
#   data/datasets/test/benchmark-dryrun-requested/
#   data/datasets/test/benchmark-dryrun/
#   (và các thư mục benchmark-dryrun-* khác)
#
# Script này:
#   1. Đọc từng file JSONL test dataset
#   2. Chạy inference qua pipeline A-MSB-GRPO (hoặc đánh giá offline)
#   3. Tính metrics (pass@1, pass@k, accuracy)
#   4. Xuất kết quả ra compare/A-MSB-GRPO/eval_results/
#
# Cách dùng:
#   chmod +x test.sh
#   ./test.sh                       # Chạy evaluation trên tất cả datasets
#   ./test.sh --dataset math500     # Chỉ chạy 1 dataset cụ thể
#   ./test.sh --offline             # Đánh giá offline (không cần GPU/model)
#   ./test.sh --dry-run             # Chỉ in lệnh, không chạy
# ==============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEST_DATA_ROOT="${REPO_ROOT}/data/datasets/test"
EVAL_OUTPUT_DIR="${SCRIPT_DIR}/eval_results"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────────────────────────────────────
# Flags
# ──────────────────────────────────────────────────────────────────────────────
TARGET_DATASET=""
OFFLINE_MODE=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     TARGET_DATASET="$2"; shift 2 ;;
        --offline)     OFFLINE_MODE=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --verbose|-v)  VERBOSE=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME    Chỉ chạy dataset cụ thể (e.g., math500, gsm8k)"
            echo "  --offline         Đánh giá offline từ JSONL có sẵn (không cần model)"
            echo "  --dry-run         Chỉ in lệnh, không thực thi"
            echo "  --verbose, -v     Hiện chi tiết từng câu hỏi"
            echo "  -h, --help        Hiển thị trợ giúp"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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
echo "║       A-MSB-GRPO Benchmark Evaluation                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_info "Timestamp:    ${TIMESTAMP}"
log_info "Test data:    ${TEST_DATA_ROOT}"
log_info "Output dir:   ${EVAL_OUTPUT_DIR}"
log_info "Offline mode: ${OFFLINE_MODE}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Tạo evaluation script (inline Python)
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "${EVAL_OUTPUT_DIR}"

cat > "${EVAL_SCRIPT}" << 'PYTHON_EVAL_SCRIPT'
#!/usr/bin/env python3
"""
A-MSB-GRPO Benchmark Evaluator
================================
Đọc các JSONL benchmark files, tính metrics (accuracy, pass@k),
và so sánh giữa các method.

JSONL Schema:
  - prompt/question: câu hỏi
  - answer/gold_answer: đáp án đúng
  - pred_answer: đáp án model dự đoán
  - label: True/False (đã đánh giá)
  - method: tên phương pháp (vanilla, amsb, mgrpo, seed, ...)
  - dataset: tên dataset
  - question_id: ID câu hỏi
  - generation_id: ID lần sinh (cho pass@k)
"""

import json
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Tính pass@k theo công thức chính xác (unbiased estimator).

    Args:
        n: tổng số mẫu sinh cho 1 câu hỏi
        c: số mẫu đúng
        k: giá trị k

    Returns:
        float: pass@k ∈ [0, 1]
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_dataset(rows: list, dataset_name: str) -> dict:
    """
    Đánh giá 1 dataset: tính accuracy và pass@k.

    Args:
        rows: list of dicts từ JSONL
        dataset_name: tên dataset

    Returns:
        dict chứa metrics
    """
    # Nhóm theo question_id
    questions = defaultdict(list)
    for row in rows:
        qid = row.get("question_id", 0)
        questions[qid].append(row)

    total_correct = sum(1 for r in rows if r.get("label", False))
    total_rows = len(rows)
    num_questions = len(questions)

    # Tính pass@k cho từng câu hỏi
    k_values = [1, 2, 4, 8]
    pass_at_k_results = {}

    for k in k_values:
        scores = []
        for qid, gens in questions.items():
            n = len(gens)
            c = sum(1 for g in gens if g.get("label", False))
            if n >= k:
                scores.append(pass_at_k(n, c, k))
        if scores:
            pass_at_k_results[f"pass@{k}"] = sum(scores) / len(scores)

    # Accuracy (pass@1 equivalent)
    accuracy = total_correct / total_rows if total_rows > 0 else 0.0

    return {
        "dataset": dataset_name,
        "total_records": total_rows,
        "total_questions": num_questions,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k_results,
        "method": rows[0].get("method", "unknown") if rows else "unknown",
    }


def evaluate_benchmark_dir(bench_dir: str) -> dict:
    """
    Đánh giá một thư mục benchmark (chứa nhiều dataset JSONL).

    Returns:
        dict chứa tổng hợp kết quả
    """
    bench_path = Path(bench_dir)
    bench_name = bench_path.name
    results = {"benchmark": bench_name, "datasets": {}, "overall": {}}

    method = "unknown"
    all_correct = 0
    all_total = 0

    for jsonl_file in sorted(bench_path.glob("*.jsonl")):
        dataset_name = jsonl_file.stem
        rows = load_jsonl(str(jsonl_file))

        if not rows:
            continue

        metrics = evaluate_dataset(rows, dataset_name)
        results["datasets"][dataset_name] = metrics
        method = metrics["method"]

        all_correct += metrics["total_correct"]
        all_total += metrics["total_records"]

    results["overall"] = {
        "method": method,
        "total_correct": all_correct,
        "total_records": all_total,
        "overall_accuracy": all_correct / all_total if all_total > 0 else 0.0,
    }

    return results


def print_results(results: dict, verbose: bool = False):
    """Pretty-print evaluation results."""
    bench = results["benchmark"]
    overall = results["overall"]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {bench}")
    print(f"  Method:    {overall['method']}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Records':>8} {'Correct':>8} {'Acc':>8}  {'pass@1':>8} {'pass@2':>8} {'pass@4':>8} {'pass@8':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for ds_name, metrics in results["datasets"].items():
        pk = metrics["pass_at_k"]
        print(f"{ds_name:<20} {metrics['total_records']:>8d} {metrics['total_correct']:>8d} "
              f"{metrics['accuracy']:>7.1%}  "
              f"{pk.get('pass@1', 0):>7.4f} {pk.get('pass@2', 0):>7.4f} "
              f"{pk.get('pass@4', 0):>7.4f} {pk.get('pass@8', 0):>7.4f}")

    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'OVERALL':<20} {overall['total_records']:>8d} {overall['total_correct']:>8d} "
          f"{overall['overall_accuracy']:>7.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="A-MSB-GRPO Benchmark Evaluator")
    parser.add_argument("--test-root", required=True, help="Root dir of test datasets")
    parser.add_argument("--output-dir", required=True, help="Output dir for results")
    parser.add_argument("--dataset", default="", help="Filter by dataset name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all methods side-by-side")
    args = parser.parse_args()

    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả benchmark directories
    bench_dirs = sorted([
        d for d in test_root.iterdir()
        if d.is_dir() and any(d.glob("*.jsonl"))
    ])

    if not bench_dirs:
        print(f"No benchmark directories with JSONL files found in: {test_root}")
        sys.exit(1)

    all_results = []

    for bench_dir in bench_dirs:
        results = evaluate_benchmark_dir(str(bench_dir))
        all_results.append(results)
        print_results(results, verbose=args.verbose)

        # Lưu kết quả JSON cho từng benchmark
        out_file = output_dir / f"{bench_dir.name}_eval.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Comparison Table ──
    if args.compare and len(all_results) > 1:
        print("\n" + "=" * 80)
        print("  CROSS-METHOD COMPARISON (Overall Accuracy)")
        print("=" * 80)

        # Collect all dataset names
        all_datasets = set()
        for r in all_results:
            all_datasets.update(r["datasets"].keys())
        all_datasets = sorted(all_datasets)

        # Header
        method_names = [r["overall"]["method"] + f" ({r['benchmark']})" for r in all_results]
        header = f"{'Dataset':<20}"
        for name in method_names:
            header += f" {name:>20}"
        print(header)
        print("-" * len(header))

        # Rows
        for ds in all_datasets:
            row = f"{ds:<20}"
            for r in all_results:
                if ds in r["datasets"]:
                    acc = r["datasets"][ds]["accuracy"]
                    row += f" {acc:>19.1%}"
                else:
                    row += f" {'N/A':>20}"
            print(row)

        # Overall
        print("-" * len(header))
        overall_row = f"{'OVERALL':<20}"
        for r in all_results:
            acc = r["overall"]["overall_accuracy"]
            overall_row += f" {acc:>19.1%}"
        print(overall_row)
        print()

    # Lưu tổng hợp
    summary_file = output_dir / f"eval_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": all_results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYTHON_EVAL_SCRIPT

chmod +x "${EVAL_SCRIPT}"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 0: Sinh kết quả dự đoán từ mô hình (vLLM Generation)
# ──────────────────────────────────────────────────────────────────────────────
LATEST_CKPT="${SCRIPT_DIR}/checkpoints/latest"
NEW_BENCHMARK_DIR="${TEST_DATA_ROOT}/benchmark-amsb-latest"
TEST_PARQUET="${REPO_ROOT}/data/datasets/requested/test/combined_test.parquet"

if [[ "$OFFLINE_MODE" == "false" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "PHASE 0: Running Generation with latest checkpoint..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ ! -d "$LATEST_CKPT" ]]; then
        log_error "Không tìm thấy checkpoint mới nhất tại: ${LATEST_CKPT}"
        log_warn "Vui lòng train model trước khi chạy test, hoặc dùng cờ --offline"
        exit 1
    fi
    
    mkdir -p "$NEW_BENCHMARK_DIR"
    
    # Số lượng câu trả lời sinh ra cho pass@k (k=8)
    PASS_K=8
    
    GEN_CMD=(
        python3 "${SCRIPT_DIR}/run_test.py"
        --test-parquet "${TEST_PARQUET}"
        --lora-dir "${LATEST_CKPT}"
        --output-dir "${NEW_BENCHMARK_DIR}"
        --n-samples "${PASS_K}"
    )
    
    log_info "Command: ${GEN_CMD[*]}"
    run_cmd "${GEN_CMD[@]}"
    
    if [[ $? -ne 0 ]]; then
        log_error "Generation thất bại! Hủy evaluation."
        exit 1
    fi
    log_success "Đã sinh xong dữ liệu đưa vào: ${NEW_BENCHMARK_DIR}"
    echo ""
else
    log_info "Skip Generation Mode (--offline). Đang đánh giá trên JSONL có sẵn."
fi

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: Xác định các benchmark directories
# ──────────────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "PHASE 1: Scanning test datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BENCH_DIRS=()
for dir in "${TEST_DATA_ROOT}"/*/; do
    if ls "${dir}"*.jsonl 1>/dev/null 2>&1; then
        dirname=$(basename "$dir")
        # Filter by target dataset if specified
        if [[ -n "$TARGET_DATASET" ]]; then
            if ls "${dir}"*"${TARGET_DATASET}"*.jsonl 1>/dev/null 2>&1; then
                BENCH_DIRS+=("$dir")
                log_info "  Found: ${dirname} (filtered for ${TARGET_DATASET})"
            fi
        else
            BENCH_DIRS+=("$dir")
            jsonl_count=$(ls "${dir}"*.jsonl 2>/dev/null | wc -l)
            log_info "  Found: ${dirname} (${jsonl_count} JSONL files)"
        fi
    fi
done

if [[ ${#BENCH_DIRS[@]} -eq 0 ]]; then
    log_error "No benchmark directories found!"
    exit 1
fi

log_success "Found ${#BENCH_DIRS[@]} benchmark suites"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: Run Evaluation
# ──────────────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "PHASE 2: Running evaluation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EVAL_LOG="${EVAL_OUTPUT_DIR}/eval_${TIMESTAMP}.log"

EVAL_CMD=(
    python3 "${EVAL_SCRIPT}"
    --test-root "${TEST_DATA_ROOT}"
    --output-dir "${EVAL_OUTPUT_DIR}"
    --compare
)

if [[ -n "$TARGET_DATASET" ]]; then
    EVAL_CMD+=(--dataset "$TARGET_DATASET")
fi

if [[ "$VERBOSE" == "true" ]]; then
    EVAL_CMD+=(--verbose)
fi

log_info "Command: ${EVAL_CMD[*]}"
echo ""

run_cmd "${EVAL_CMD[@]}" 2>&1 | tee "$EVAL_LOG"

EVAL_EXIT=${PIPESTATUS[0]}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$EVAL_EXIT" -eq 0 ]]; then
    log_success "Evaluation hoàn tất!"
    log_info "Results:  ${EVAL_OUTPUT_DIR}/"
    log_info "Log:      ${EVAL_LOG}"
    echo ""
    log_info "Các file kết quả:"
    ls -la "${EVAL_OUTPUT_DIR}"/*.json 2>/dev/null | while read -r line; do
        echo "  $line"
    done
else
    log_error "Evaluation thất bại! Exit code: ${EVAL_EXIT}"
    log_error "Xem log: ${EVAL_LOG}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$EVAL_EXIT"
