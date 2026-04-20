# Compare-Level Pipeline Runner

This folder now has a single entrypoint to orchestrate multi-repo train/eval/metrics/chart steps.

## Files

- `run_compare_pipeline.py`: orchestrates stages (`plan`, `validate-datasets`, `train`, `eval`, `collect`, `chart`, `all`)
- `pipeline.compare.yaml`: run registry for `open-r1`, `NGRPO`, `DIVA-GRPO`, `DeepSeek-Math`
- `datasets.compare.manifest.yaml`: canonical dataset contract
- `datasets.compare.lock.json`: optional row/hash lock metadata

## Quick Start

Run from `compare/`.

```powershell
python run_compare_pipeline.py --stage plan
```

Validate dataset files and schema:

```powershell
python run_compare_pipeline.py --stage validate-datasets
```

Run full flow with command preview only:

```powershell
python run_compare_pipeline.py --stage all --dry-run
```

Run real train/eval for enabled runs:

```powershell
python run_compare_pipeline.py --stage all
```

Collect metrics only:

```powershell
python run_compare_pipeline.py --stage collect
```

Render charts from normalized metrics:

```powershell
python run_compare_pipeline.py --stage chart --metric pass@1
```

## Enabling More Runs

`pipeline.compare.yaml` keeps `NGRPO` and `DIVA-GRPO` entries as templates (`enabled: false`) because those scripts require environment/path customization.

To enable them:

1. Set `enabled: true` in the run entry.
2. Update `train_command`, `eval_command`, and metric collector paths.
3. Re-run `plan` first, then `all`.

## Output Artifacts

By default artifacts are written to:

- `artifacts/compare/normalized_metrics.json`
- `artifacts/compare/normalized_metrics.csv`
- `artifacts/compare/chart_by_dataset_<metric>.png`
- `artifacts/compare/chart_overall_<metric>.png`

## Notes

- Dataset validation supports parquet column checks when `pyarrow` is available.
- Chart rendering requires `matplotlib`.
- For strict CI behavior, add `--strict`.
