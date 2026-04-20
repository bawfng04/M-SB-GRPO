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
