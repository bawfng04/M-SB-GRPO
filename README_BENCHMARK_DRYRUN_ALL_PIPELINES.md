# Benchmark Dryrun Guide (Run From Repo Root)

This guide shows how to run benchmark dryrun for all enabled pipelines when your shell is at repo root:

`D:\Projects\M-SB-GRPO`

## 1) What "all pipelines" means here

The compare orchestrator reads runs from:

- `compare/pipeline.compare.yaml`

Current enabled runs:

- `openr1_vanilla`
- `openr1_mgrpo`
- `openr1_seed`
- `openr1_amsb`
- `deepseek_eval_snapshot`

Notes:

- `ngrpo_template` and `diva_template` are disabled by default (`enabled: false`).
- If you want those included, enable and configure them first in `compare/pipeline.compare.yaml`.

## 2) Quick path sanity check (important)

From repo root in PowerShell:

```powershell
Test-Path .\compare\open-r1\.venv\Scripts\python.exe
```

Expected output: `True`

If your prompt is `>>>`, you are inside Python REPL, not PowerShell. Exit REPL first (`exit()` or `Ctrl+Z`, then Enter).

`.\compare\open-r1\.venv\Scripts\python.exe`

## 3) Plan first (recommended)

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage plan
```

This prints train/eval commands for each selected run.

## 4) Run full dryrun flow for all enabled pipelines

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage all
```

`--stage all` executes:

1. `plan`
2. `validate-datasets`
3. `train`
4. `eval`
5. `collect`
6. `chart`

By default (without `--strict`), per-run train/eval failures are logged but the orchestrator can continue to later stages.

## 5) Preview only (no train/eval execution)

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage all --dry-run
```

## 6) Benchmark-only mode (skip train stage)

Use this if dryrun train outputs are already present and you only want benchmark + aggregation:

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage eval
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage collect
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage chart --metric pass@1
```

## 7) Strict mode (fail on warnings where possible)

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage all --strict
```

## 8) Run subset of pipelines

Example: run only Open-R1 variants.

```powershell
& ".\compare\open-r1\.venv\Scripts\python.exe" ".\compare\run_compare_pipeline.py" --config ".\compare\pipeline.compare.yaml" --stage all --runs openr1_vanilla,openr1_mgrpo,openr1_seed,openr1_amsb
```

## 9) Output artifacts

Main compare outputs:

- `compare/artifacts/compare/normalized_metrics.json`
- `compare/artifacts/compare/normalized_metrics.csv`
- `compare/artifacts/compare/chart_by_dataset_pass@1.png`
- `compare/artifacts/compare/chart_overall_pass@1.png`

Open-R1 benchmark dryrun summaries:

- `compare/open-r1/data/benchmark-dryrun-vanilla/summary.json`
- `compare/open-r1/data/benchmark-dryrun-mgrpo/summary.json`
- `compare/open-r1/data/benchmark-dryrun-seed/summary.json`
- `compare/open-r1/data/benchmark-dryrun-amsb/summary.json`

## 10) Troubleshooting

If PowerShell says executable not recognized:

1. Re-check the exact path in section 2.
2. Use an absolute command:

```powershell
& "D:\Projects\M-SB-GRPO\compare\open-r1\.venv\Scripts\python.exe" "D:\Projects\M-SB-GRPO\compare\run_compare_pipeline.py" --config "D:\Projects\M-SB-GRPO\compare\pipeline.compare.yaml" --stage plan
```

If dependency errors appear (example: `No module named trl`), install missing packages into `compare/open-r1/.venv` before re-running.
