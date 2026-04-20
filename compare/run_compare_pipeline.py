#!/usr/bin/env python3
"""Unified compare-level pipeline runner for train/eval/metrics/chart stages."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _resolve_path(base_dir: Path, maybe_relative: str | None) -> Path:
    if not maybe_relative:
        return base_dir
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _unbiased_pass_at_k(total_generations: int, num_correct: int, k: int) -> float:
    if total_generations <= 0:
        return 0.0
    if num_correct <= 0:
        return 0.0
    k = min(k, total_generations)
    if total_generations - num_correct < k:
        return 1.0

    probability_all_fail = 1.0
    for i in range(k):
        probability_all_fail *= (total_generations - num_correct - i) / (
            total_generations - i
        )
    return 1.0 - probability_all_fail


def _append_metric(
    sink: list[dict[str, Any]],
    *,
    run_id: str,
    repo: str,
    method: str,
    dataset: str,
    metric: str,
    value: float,
    source: str,
    n_samples: int | None = None,
) -> None:
    sink.append(
        {
            "run_id": run_id,
            "repo": repo,
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "value": float(value),
            "n_samples": n_samples,
            "source": source,
        }
    )


def _collect_openr1_dryrun_summary(
    sink: list[dict[str, Any]],
    *,
    run_cfg: dict[str, Any],
    collector_cfg: dict[str, Any],
    file_path: Path,
) -> None:
    payload = _load_json(file_path)

    run_id = str(run_cfg["id"])
    repo = str(run_cfg.get("repo", ""))
    method = str(run_cfg.get("method", payload.get("method", "unknown")))
    dataset = str(payload.get("train_split", "train"))

    train_samples = payload.get("train_samples")
    if _is_number(train_samples):
        _append_metric(
            sink,
            run_id=run_id,
            repo=repo,
            method=method,
            dataset=dataset,
            metric="train_samples",
            value=float(train_samples),
            n_samples=int(train_samples),
            source=str(file_path),
        )

    method_diagnostics = payload.get("method_diagnostics", {})
    if isinstance(method_diagnostics, dict):
        for key, raw_value in method_diagnostics.items():
            if not _is_number(raw_value):
                continue
            _append_metric(
                sink,
                run_id=run_id,
                repo=repo,
                method=method,
                dataset=dataset,
                metric=f"method/{key}",
                value=float(raw_value),
                n_samples=int(train_samples) if _is_number(train_samples) else None,
                source=str(file_path),
            )


def _collect_openr1_benchmark_summary(
    sink: list[dict[str, Any]],
    *,
    run_cfg: dict[str, Any],
    collector_cfg: dict[str, Any],
    file_path: Path,
) -> None:
    payload = _load_json(file_path)
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return

    run_id = str(run_cfg["id"])
    repo = str(run_cfg.get("repo", ""))
    method = str(run_cfg.get("method", payload.get("method", "unknown")))

    for dataset_name, summary in datasets.items():
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("metrics", {})
        n_samples = summary.get("questions")
        if not isinstance(metrics, dict):
            continue
        for metric_name, raw_value in metrics.items():
            if not _is_number(raw_value):
                continue
            _append_metric(
                sink,
                run_id=run_id,
                repo=repo,
                method=method,
                dataset=str(dataset_name),
                metric=str(metric_name),
                value=float(raw_value),
                n_samples=int(n_samples) if _is_number(n_samples) else None,
                source=str(file_path),
            )


def _collect_jsonl_accuracy(
    sink: list[dict[str, Any]],
    *,
    run_cfg: dict[str, Any],
    collector_cfg: dict[str, Any],
    files: list[Path],
) -> None:
    run_id = str(run_cfg["id"])
    repo = str(run_cfg.get("repo", ""))
    method = str(run_cfg.get("method", "unknown"))

    for file_path in files:
        labels_by_dataset: dict[str, list[bool]] = defaultdict(list)
        labels_by_dataset_question: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                dataset_name = row.get("dataset") or row.get("dataset_name")
                if not dataset_name:
                    configured_dataset = collector_cfg.get("dataset")
                    if configured_dataset:
                        dataset_name = configured_dataset
                    else:
                        dataset_name = file_path.stem
                dataset_name = str(dataset_name)

                label = _parse_bool_label(row.get("label"))
                if label is None:
                    continue

                labels_by_dataset[dataset_name].append(label)

                question_id = row.get("question_id")
                if question_id is not None:
                    labels_by_dataset_question[dataset_name][str(question_id)].append(
                        label
                    )

        for dataset_name, labels in labels_by_dataset.items():
            if not labels:
                continue

            n_samples = len(labels)
            accuracy = sum(1.0 for item in labels if item) / n_samples
            _append_metric(
                sink,
                run_id=run_id,
                repo=repo,
                method=method,
                dataset=dataset_name,
                metric="accuracy",
                value=accuracy,
                n_samples=n_samples,
                source=str(file_path),
            )

            by_question = labels_by_dataset_question.get(dataset_name, {})
            if not by_question:
                continue

            pass_scores: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
            for _, question_labels in by_question.items():
                total_generations = len(question_labels)
                num_correct = sum(1 for item in question_labels if item)
                for k in pass_scores:
                    score = _unbiased_pass_at_k(total_generations, num_correct, k)
                    pass_scores[k].append(score)

            for k, values in pass_scores.items():
                if not values:
                    continue
                _append_metric(
                    sink,
                    run_id=run_id,
                    repo=repo,
                    method=method,
                    dataset=dataset_name,
                    metric=f"pass@{k}",
                    value=sum(values) / len(values),
                    n_samples=len(values),
                    source=str(file_path),
                )


def _collect_deepseek_evaluation_results(
    sink: list[dict[str, Any]],
    *,
    run_cfg: dict[str, Any],
    collector_cfg: dict[str, Any],
    file_path: Path,
) -> None:
    payload = _load_json(file_path)

    repo = str(run_cfg.get("repo", ""))
    base_method = str(run_cfg.get("method", "deepseek-eval"))
    base_run_id = str(run_cfg["id"])

    for model_name, dataset_map in payload.items():
        if not isinstance(dataset_map, dict):
            continue
        run_id = f"{base_run_id}:{model_name}"
        method = f"{base_method}:{model_name}"

        for dataset_name, mode_map in dataset_map.items():
            if not isinstance(mode_map, dict):
                continue
            for mode_name, metrics in mode_map.items():
                if not isinstance(metrics, dict):
                    continue

                n_samples = metrics.get("n_samples")
                for metric_name, raw_value in metrics.items():
                    if metric_name == "n_samples":
                        continue
                    if not _is_number(raw_value):
                        continue

                    metric = f"{mode_name}/{metric_name}"
                    _append_metric(
                        sink,
                        run_id=run_id,
                        repo=repo,
                        method=method,
                        dataset=str(dataset_name),
                        metric=metric,
                        value=float(raw_value),
                        n_samples=int(n_samples) if _is_number(n_samples) else None,
                        source=str(file_path),
                    )

                    if metric_name == "accuracy":
                        _append_metric(
                            sink,
                            run_id=run_id,
                            repo=repo,
                            method=method,
                            dataset=f"{dataset_name}::{mode_name}",
                            metric="accuracy",
                            value=float(raw_value),
                            n_samples=int(n_samples) if _is_number(n_samples) else None,
                            source=str(file_path),
                        )


def _collector_files(repo_dir: Path, collector_cfg: dict[str, Any]) -> list[Path]:
    if "path" in collector_cfg and collector_cfg["path"]:
        return [_resolve_path(repo_dir, str(collector_cfg["path"]))]

    pattern = collector_cfg.get("glob")
    if not pattern:
        return []

    search_pattern = str(_resolve_path(repo_dir, str(pattern)))
    return [Path(item).resolve() for item in glob.glob(search_pattern, recursive=True)]


def collect_metrics(
    *,
    workspace_root: Path,
    runs: list[dict[str, Any]],
    strict: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for run_cfg in runs:
        repo_dir = _resolve_path(workspace_root, str(run_cfg.get("repo", ".")))
        collectors = run_cfg.get("metric_collectors", [])
        if not isinstance(collectors, list):
            warnings.append(
                f"Run '{run_cfg.get('id')}' has invalid metric_collectors format"
            )
            continue

        for collector_cfg in collectors:
            if not isinstance(collector_cfg, dict):
                continue
            kind = str(collector_cfg.get("kind", "")).strip()
            if not kind:
                continue

            files = _collector_files(repo_dir, collector_cfg)
            if not files:
                warnings.append(
                    f"Collector '{kind}' in run '{run_cfg.get('id')}' matched no files"
                )
                continue

            for file_path in files:
                if not file_path.exists():
                    warnings.append(
                        f"Collector '{kind}' in run '{run_cfg.get('id')}' missing file: {file_path}"
                    )
                    continue

                try:
                    if kind == "openr1_dryrun_summary":
                        _collect_openr1_dryrun_summary(
                            records,
                            run_cfg=run_cfg,
                            collector_cfg=collector_cfg,
                            file_path=file_path,
                        )
                    elif kind == "openr1_benchmark_summary":
                        _collect_openr1_benchmark_summary(
                            records,
                            run_cfg=run_cfg,
                            collector_cfg=collector_cfg,
                            file_path=file_path,
                        )
                    elif kind == "jsonl_accuracy":
                        _collect_jsonl_accuracy(
                            records,
                            run_cfg=run_cfg,
                            collector_cfg=collector_cfg,
                            files=[file_path],
                        )
                    elif kind == "deepseek_evaluation_results":
                        _collect_deepseek_evaluation_results(
                            records,
                            run_cfg=run_cfg,
                            collector_cfg=collector_cfg,
                            file_path=file_path,
                        )
                    else:
                        warnings.append(
                            f"Unsupported collector kind '{kind}' in run '{run_cfg.get('id')}'"
                        )
                except Exception as exc:
                    warnings.append(
                        f"Collector '{kind}' in run '{run_cfg.get('id')}' failed on '{file_path}': {exc}"
                    )

    if strict and warnings:
        for warning in warnings:
            print(f"[collect][warning] {warning}")
        raise RuntimeError("Metric collection failed in strict mode")

    return records, warnings


def _write_metrics_artifacts(
    records: list[dict[str, Any]], artifacts_dir: Path
) -> tuple[Path, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    normalized = sorted(
        records,
        key=lambda row: (
            row["metric"],
            row["dataset"],
            row["run_id"],
            row["repo"],
            row["method"],
        ),
    )

    json_path = artifacts_dir / "normalized_metrics.json"
    csv_path = artifacts_dir / "normalized_metrics.csv"

    json_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    fieldnames = [
        "run_id",
        "repo",
        "method",
        "dataset",
        "metric",
        "value",
        "n_samples",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized:
            writer.writerow(row)

    return json_path, csv_path


def _choose_metric(
    records: list[dict[str, Any]], requested_metric: str | None
) -> str | None:
    available = {str(row["metric"]) for row in records}
    if not available:
        return None

    if requested_metric and requested_metric in available:
        return requested_metric

    priorities = [
        "pass@1",
        "accuracy",
        "cot/accuracy",
        "tool/accuracy",
        "pass@4",
    ]
    for item in priorities:
        if item in available:
            return item

    return sorted(available)[0]


def generate_charts(
    *,
    records: list[dict[str, Any]],
    artifacts_dir: Path,
    metric: str | None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for chart generation. Install with 'pip install matplotlib'."
        ) from exc

    chosen_metric = _choose_metric(records, metric)
    if not chosen_metric:
        raise RuntimeError("No metrics available for chart generation")

    metric_rows = [row for row in records if row["metric"] == chosen_metric]
    if not metric_rows:
        raise RuntimeError(f"No rows found for metric '{chosen_metric}'")

    by_dataset_run: dict[str, dict[str, float]] = defaultdict(dict)
    for row in metric_rows:
        dataset = str(row["dataset"])
        run_id = str(row["run_id"])
        by_dataset_run[dataset][run_id] = float(row["value"])

    datasets = sorted(by_dataset_run.keys())
    runs = sorted(
        {run for mapping in by_dataset_run.values() for run in mapping.keys()}
    )

    if not datasets or not runs:
        raise RuntimeError("Not enough data to render charts")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    # Chart 1: grouped by dataset.
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 6))
    x_positions = list(range(len(datasets)))
    bar_width = 0.8 / max(len(runs), 1)

    for idx, run_id in enumerate(runs):
        values = [by_dataset_run[dataset].get(run_id, 0.0) for dataset in datasets]
        offsets = [x + (idx - (len(runs) - 1) / 2.0) * bar_width for x in x_positions]
        ax.bar(offsets, values, width=bar_width, label=run_id)

    ax.set_title(f"Compare Metrics by Dataset ({chosen_metric})")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(chosen_metric)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    dataset_chart_path = (
        artifacts_dir / f"chart_by_dataset_{chosen_metric.replace('/', '_')}.png"
    )
    fig.savefig(dataset_chart_path, dpi=180)
    plt.close(fig)
    output_paths.append(dataset_chart_path)

    # Chart 2: overall mean by run.
    mean_by_run: dict[str, float] = {}
    for run_id in runs:
        values = [
            by_dataset_run[dataset][run_id]
            for dataset in datasets
            if run_id in by_dataset_run[dataset]
        ]
        mean_by_run[run_id] = (sum(values) / len(values)) if values else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.0), 5))
    run_positions = list(range(len(runs)))
    run_values = [mean_by_run[run_id] for run_id in runs]

    ax.bar(run_positions, run_values, width=0.6)
    ax.set_title(f"Overall Mean Score ({chosen_metric})")
    ax.set_xlabel("Run")
    ax.set_ylabel(chosen_metric)
    ax.set_xticks(run_positions)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    overall_chart_path = (
        artifacts_dir / f"chart_overall_{chosen_metric.replace('/', '_')}.png"
    )
    fig.savefig(overall_chart_path, dpi=180)
    plt.close(fig)
    output_paths.append(overall_chart_path)

    return output_paths


def _print_commands(workspace_root: Path, runs: list[dict[str, Any]]) -> None:
    for run_cfg in runs:
        run_id = run_cfg.get("id")
        repo_dir = _resolve_path(workspace_root, str(run_cfg.get("repo", ".")))
        train_command = run_cfg.get("train_command")
        eval_command = run_cfg.get("eval_command")

        print(f"[plan] run={run_id} repo={repo_dir}")
        if train_command:
            print(f"  train: {train_command}")
        if eval_command:
            print(f"  eval : {eval_command}")


def _run_command(command: str, *, cwd: Path, dry_run: bool) -> int:
    print(f"[exec] cwd={cwd}")
    print(f"[exec] command={command}")
    if dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(cwd), shell=True, check=False)
    return int(completed.returncode)


def run_stage_commands(
    *,
    workspace_root: Path,
    runs: list[dict[str, Any]],
    stage_name: str,
    dry_run: bool,
    strict: bool,
) -> list[str]:
    failures: list[str] = []

    command_key = "train_command" if stage_name == "train" else "eval_command"
    for run_cfg in runs:
        command = run_cfg.get(command_key)
        if not command:
            continue

        repo_dir = _resolve_path(workspace_root, str(run_cfg.get("repo", ".")))
        return_code = _run_command(str(command), cwd=repo_dir, dry_run=dry_run)
        if return_code != 0:
            message = (
                f"Run '{run_cfg.get('id')}' failed at stage '{stage_name}' "
                f"with exit code {return_code}"
            )
            failures.append(message)
            print(f"[stage][error] {message}")
            if strict:
                break

    return failures


def validate_datasets(
    *,
    workspace_root: Path,
    manifest_path: Path,
    lock_path: Path,
    strict: bool,
    verify_hash: bool,
) -> tuple[list[str], list[str]]:
    ok_lines: list[str] = []
    warning_lines: list[str] = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing dataset lock: {lock_path}")

    manifest = _load_yaml(manifest_path)
    lock = _load_json(lock_path)

    datasets = manifest.get("datasets", [])
    locked = lock.get("locked_datasets", {})

    if not isinstance(datasets, list):
        raise ValueError("dataset manifest field 'datasets' must be a list")
    if not isinstance(locked, dict):
        raise ValueError("dataset lock field 'locked_datasets' must be a mapping")

    for item in datasets:
        if not isinstance(item, dict):
            continue

        dataset_id = str(item.get("id", ""))
        split = str(item.get("split", "unknown"))
        relative_path = str(item.get("path", ""))
        file_format = str(item.get("format", "")).lower()
        required_columns = item.get("required_columns", [])

        if not dataset_id or not relative_path:
            warning_lines.append(
                f"Invalid manifest entry (missing id/path): {json.dumps(item, ensure_ascii=True)}"
            )
            continue

        file_path = _resolve_path(workspace_root, relative_path)
        if not file_path.exists():
            warning_lines.append(f"Missing dataset file: {file_path}")
            continue

        lock_entry = locked.get(dataset_id, {})
        if not isinstance(lock_entry, dict):
            lock_entry = {}

        row_count: int | None = None
        if file_format == "parquet":
            try:
                import pyarrow.parquet as pq
            except Exception as exc:
                message = (
                    "pyarrow is required for parquet schema validation. "
                    "Install with 'pip install pyarrow'."
                )
                if strict:
                    raise RuntimeError(message) from exc
                warning_lines.append(message)
                continue

            table = pq.read_table(file_path)
            row_count = table.num_rows
            columns = set(table.column_names)
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                warning_lines.append(
                    f"Missing columns in {dataset_id}/{split}: {missing_columns}"
                )

        expected_row_count = lock_entry.get("row_count")
        if expected_row_count is not None and row_count is not None:
            if int(expected_row_count) != int(row_count):
                warning_lines.append(
                    f"Row count mismatch for {dataset_id}: expected={expected_row_count} actual={row_count}"
                )

        expected_sha = lock_entry.get("sha256")
        if expected_sha:
            if verify_hash:
                actual_sha = _sha256(file_path)
                if actual_sha != expected_sha:
                    warning_lines.append(
                        f"SHA256 mismatch for {dataset_id}: expected={expected_sha} actual={actual_sha}"
                    )
            else:
                warning_lines.append(
                    f"SHA256 exists for {dataset_id} but --verify-hash was not enabled"
                )

        ok_lines.append(
            f"Validated dataset {dataset_id}/{split} at {file_path}"
            + (f" (rows={row_count})" if row_count is not None else "")
        )

    if strict and warning_lines:
        raise RuntimeError("Dataset validation failed in strict mode")

    return ok_lines, warning_lines


def _select_runs(
    all_runs: list[dict[str, Any]],
    selected_ids: list[str],
    include_disabled: bool,
) -> list[dict[str, Any]]:
    selected_set = {item.strip() for item in selected_ids if item.strip()}

    chosen: list[dict[str, Any]] = []
    for run_cfg in all_runs:
        run_id = str(run_cfg.get("id", ""))
        enabled = bool(run_cfg.get("enabled", True))

        if selected_set and run_id not in selected_set:
            continue
        if not include_disabled and not enabled:
            continue

        chosen.append(run_cfg)

    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compare-wide training/evaluation pipeline"
    )
    parser.add_argument(
        "--config",
        default="pipeline.compare.yaml",
        help="Path to pipeline compare YAML config",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "validate-datasets",
            "train",
            "eval",
            "collect",
            "chart",
            "all",
        ],
        default="all",
        help="Stage to execute",
    )
    parser.add_argument(
        "--runs",
        default="",
        help="Comma-separated run IDs to execute. Empty means all enabled runs.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include disabled runs from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing train/eval stages",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on warnings/errors where possible",
    )
    parser.add_argument(
        "--metric",
        default="",
        help="Preferred metric for chart stage (for example: pass@1 or accuracy)",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Verify file hashes from lock file during dataset validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Missing config: {config_path}", file=sys.stderr)
        return 1

    config = _load_yaml(config_path)
    config_dir = config_path.parent
    workspace_root = _resolve_path(config_dir, str(config.get("workspace_root", ".")))
    artifacts_dir = _resolve_path(
        workspace_root, str(config.get("artifacts_dir", "artifacts/compare"))
    )

    all_runs = config.get("runs", [])
    if not isinstance(all_runs, list):
        raise ValueError("Config field 'runs' must be a list")

    selected_run_ids = [item for item in args.runs.split(",") if item.strip()]
    runs = _select_runs(all_runs, selected_run_ids, args.include_disabled)

    if not runs:
        print("No runs selected. Check --runs and enabled flags.")
        return 1

    manifest_path = _resolve_path(
        workspace_root,
        str(config.get("dataset_manifest", "datasets.compare.manifest.yaml")),
    )
    lock_path = _resolve_path(
        workspace_root, str(config.get("dataset_lock", "datasets.compare.lock.json"))
    )

    records: list[dict[str, Any]] = []

    if args.stage in {"plan", "all"}:
        print("[stage] plan")
        _print_commands(workspace_root, runs)
        if args.stage == "plan":
            return 0

    if args.stage in {"validate-datasets", "all"}:
        print("[stage] validate-datasets")
        ok_lines, warning_lines = validate_datasets(
            workspace_root=workspace_root,
            manifest_path=manifest_path,
            lock_path=lock_path,
            strict=args.strict,
            verify_hash=args.verify_hash,
        )
        for line in ok_lines:
            print(f"[dataset][ok] {line}")
        for line in warning_lines:
            print(f"[dataset][warning] {line}")
        if args.stage == "validate-datasets":
            return 0 if not (args.strict and warning_lines) else 1

    if args.stage in {"train", "all"}:
        print("[stage] train")
        failures = run_stage_commands(
            workspace_root=workspace_root,
            runs=runs,
            stage_name="train",
            dry_run=args.dry_run,
            strict=args.strict,
        )
        if failures and args.strict:
            return 1
        if args.stage == "train":
            return 0 if not failures else 1

    if args.stage in {"eval", "all"}:
        print("[stage] eval")
        failures = run_stage_commands(
            workspace_root=workspace_root,
            runs=runs,
            stage_name="eval",
            dry_run=args.dry_run,
            strict=args.strict,
        )
        if failures and args.strict:
            return 1
        if args.stage == "eval":
            return 0 if not failures else 1

    metrics_json_path: Path | None = None

    if args.stage in {"collect", "all"}:
        print("[stage] collect")
        records, warnings = collect_metrics(
            workspace_root=workspace_root,
            runs=runs,
            strict=args.strict,
        )
        for warning in warnings:
            print(f"[collect][warning] {warning}")

        metrics_json_path, metrics_csv_path = _write_metrics_artifacts(
            records, artifacts_dir
        )
        print(f"[collect] Wrote {len(records)} rows to {metrics_json_path}")
        print(f"[collect] Wrote CSV to {metrics_csv_path}")

        if args.stage == "collect":
            return 0

    if args.stage in {"chart", "all"}:
        print("[stage] chart")

        if not records:
            if metrics_json_path is None:
                metrics_json_path = artifacts_dir / "normalized_metrics.json"
            if not metrics_json_path.exists():
                print(
                    "No in-memory metrics and no normalized_metrics.json found. "
                    "Run --stage collect first.",
                    file=sys.stderr,
                )
                return 1
            loaded = json.loads(metrics_json_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("normalized_metrics.json must contain a list")
            records = [item for item in loaded if isinstance(item, dict)]

        chart_paths = generate_charts(
            records=records,
            artifacts_dir=artifacts_dir,
            metric=args.metric or None,
        )
        for path in chart_paths:
            print(f"[chart] Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
