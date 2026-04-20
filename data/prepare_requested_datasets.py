#!/usr/bin/env python3
"""Download and prepare user-requested train/test datasets into local parquet files.

Requested by project plan:
- Train: MATH (subset Level 3-5), GSM8K train
- Test: MATH500, AIME2025, OlympiadBench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import datasets
import yaml


QUESTION_KEYS = ["problem", "question", "prompt", "input", "query", "instruction"]
ANSWER_KEYS = ["answer", "solution", "final_answer", "target", "gold", "output"]


@dataclass
class DatasetSpec:
    name: str
    split_role: str  # train|test
    hf_candidates: list[str]
    config_candidates: list[Optional[str]]
    preferred_split: Optional[str]
    row_filter: Optional[Callable[[dict[str, Any]], bool]] = None


def extract_gsm8k_final_answer(answer_text: str) -> str:
    if not answer_text:
        return ""
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def pick_first(row: dict[str, Any], keys: list[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            # Keep primitive or convert dict/list to compact JSON.
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=True)
            return safe_str(value)
    return None


def load_dataset_with_fallback(spec: DatasetSpec):
    last_error = None
    for hf_id in spec.hf_candidates:
        for cfg in spec.config_candidates:
            try:
                split_names = datasets.get_dataset_split_names(hf_id, cfg)
                target_split = spec.preferred_split
                if target_split is None or target_split not in split_names:
                    for candidate in ["test", "validation", "val", "train"]:
                        if candidate in split_names:
                            target_split = candidate
                            break
                if target_split is None:
                    raise ValueError(f"No usable split for {hf_id} cfg={cfg}")

                ds = datasets.load_dataset(hf_id, cfg, split=target_split)
                return hf_id, cfg, target_split, ds
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue

    raise RuntimeError(f"Unable to load dataset for spec '{spec.name}': {last_error}")


def convert_rows(spec: DatasetSpec, hf_id: str, split_name: str, dataset_obj):
    rows = []
    skipped = 0

    for idx, row in enumerate(dataset_obj):
        if spec.row_filter is not None and not spec.row_filter(row):
            continue

        problem = pick_first(row, QUESTION_KEYS)
        solution_raw = pick_first(row, ANSWER_KEYS)

        if problem is None or solution_raw is None:
            skipped += 1
            continue

        answer_extracted = solution_raw
        if hf_id == "openai/gsm8k":
            answer_extracted = extract_gsm8k_final_answer(solution_raw)

        metadata = {
            "source_dataset": hf_id,
            "source_split": split_name,
            "index": idx,
        }

        # Keep lightweight helpful fields when present.
        for key in ["level", "type", "subject", "category", "language", "difficulty"]:
            if key in row:
                metadata[key] = row[key]

        rows.append(
            {
                "problem": safe_str(problem),
                "solution": safe_str(solution_raw),
                "answer_extracted": safe_str(answer_extracted),
                "source_dataset": hf_id,
                "source_split": split_name,
                "metadata": json.dumps(metadata, ensure_ascii=True),
            }
        )

    return rows, skipped


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def level3_to_5_filter(row: dict[str, Any]) -> bool:
    level = safe_str(row.get("level", "")).strip().lower()
    return level in {"level 3", "level 4", "level 5"}


def build_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="math_l3_l5_train",
            split_role="train",
            hf_candidates=["hendrycks/competition_math", "TianHongZXY/MATH"],
            config_candidates=[None],
            preferred_split="train",
            row_filter=level3_to_5_filter,
        ),
        DatasetSpec(
            name="gsm8k_train",
            split_role="train",
            hf_candidates=["openai/gsm8k"],
            config_candidates=["main"],
            preferred_split="train",
        ),
        DatasetSpec(
            name="math500_test",
            split_role="test",
            hf_candidates=[
                "HuggingFaceH4/MATH-500",
                "DigitalLearningGmbH/MATH-lighteval",
            ],
            config_candidates=[None],
            preferred_split="test",
        ),
        DatasetSpec(
            name="aime2025_test",
            split_role="test",
            hf_candidates=["TianHongZXY/AIME2025"],
            config_candidates=[None],
            preferred_split="test",
        ),
        DatasetSpec(
            name="olympiadbench_test",
            split_role="test",
            hf_candidates=["Hothan/OlympiadBench"],
            config_candidates=["OE_TO_maths_en_COMP"],
            preferred_split="test",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare requested train/test datasets"
    )
    parser.add_argument(
        "--output-root",
        default="data/datasets/requested",
        help="Output directory for prepared parquet files",
    )
    parser.add_argument(
        "--manifest-path",
        default="datasets/requested_datasets.manifest.yaml",
        help="Path to output manifest yaml",
    )
    parser.add_argument(
        "--lock-path",
        default="datasets/requested_datasets.lock.json",
        help="Path to output lock json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    manifest_entries = []
    lock_entries: dict[str, Any] = {}

    train_combined = []
    test_combined = []

    for spec in specs:
        hf_id, cfg, split_name, ds = load_dataset_with_fallback(spec)
        rows, skipped = convert_rows(spec, hf_id, split_name, ds)

        if not rows:
            raise RuntimeError(f"No usable rows for {spec.name} ({hf_id}/{split_name})")

        out_dir = train_dir if spec.split_role == "train" else test_dir
        out_file = out_dir / f"{spec.name}.parquet"

        prepared_ds = datasets.Dataset.from_list(rows)
        prepared_ds.to_parquet(str(out_file))

        rel_path = out_file.relative_to(repo_root).as_posix()
        file_hash = sha256_of_file(out_file)

        manifest_entries.append(
            {
                "id": spec.name,
                "split_role": spec.split_role,
                "source": "huggingface",
                "hf_id": hf_id,
                "config": cfg,
                "split": split_name,
                "size": len(rows),
                "local_path": rel_path,
                "skipped_rows_missing_mapping": skipped,
            }
        )

        lock_entries[spec.name] = {
            "source": "huggingface",
            "hf_id": hf_id,
            "config": cfg,
            "split": split_name,
            "row_count": len(rows),
            "sha256": file_hash,
            "local_path": rel_path,
        }

        if spec.split_role == "train":
            train_combined.extend(rows)
        else:
            test_combined.extend(rows)

    combined_train_path = train_dir / "combined_train.parquet"
    combined_test_path = test_dir / "combined_test.parquet"

    datasets.Dataset.from_list(train_combined).to_parquet(str(combined_train_path))
    datasets.Dataset.from_list(test_combined).to_parquet(str(combined_test_path))

    lock_entries["combined_train"] = {
        "source": "composed",
        "row_count": len(train_combined),
        "sha256": sha256_of_file(combined_train_path),
        "local_path": combined_train_path.relative_to(repo_root).as_posix(),
    }
    lock_entries["combined_test"] = {
        "source": "composed",
        "row_count": len(test_combined),
        "sha256": sha256_of_file(combined_test_path),
        "local_path": combined_test_path.relative_to(repo_root).as_posix(),
    }

    generated_dt = datetime.now(timezone.utc)
    generated_date = generated_dt.strftime("%Y-%m-%d")
    generated_iso = (
        generated_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    manifest = {
        "version": "1.0.0",
        "generated": generated_date,
        "datasets": manifest_entries,
        "combined": {
            "train": lock_entries["combined_train"]["local_path"],
            "test": lock_entries["combined_test"]["local_path"],
        },
    }

    lock = {
        "version": "1.0.0",
        "generated_at": generated_iso,
        "datasets": lock_entries,
    }

    manifest_path = (repo_root / args.manifest_path).resolve()
    lock_path = (repo_root / args.lock_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    summary = {
        "train_rows": len(train_combined),
        "test_rows": len(test_combined),
        "manifest": manifest_path.relative_to(repo_root).as_posix(),
        "lock": lock_path.relative_to(repo_root).as_posix(),
        "output_root": output_root.relative_to(repo_root).as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
