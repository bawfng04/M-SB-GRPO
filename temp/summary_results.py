import json, glob
from pathlib import Path

results_dir = Path(r"d:\Projects\M-SB-GRPO\compare\open-r1\results")

for pattern in ["AMSB_*/**/*.json", "MGRPO_*/**/*.json", "GRPO_*/**/*.json"]:
    for jf in sorted(results_dir.glob(pattern)):
        rel = jf.relative_to(results_dir)
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            results = data.get("results", {})
            for bench, metrics in results.items():
                if bench == "all":
                    p1_1 = metrics.get("math_pass@1:1_samples") or metrics.get("gpqa_pass@1:1_samples")
                    err_1 = metrics.get("math_pass@1:1_samples_stderr") or metrics.get("gpqa_pass@1:1_samples_stderr")
                    p1_4 = metrics.get("math_pass@1:4_samples") or metrics.get("gpqa_pass@1:4_samples")
                    err_4 = metrics.get("math_pass@1:4_samples_stderr") or metrics.get("gpqa_pass@1:4_samples_stderr")
                    print(f"{rel.parts[0]} -> 1-samp: {p1_1} (+/- {err_1}) | 4-samp: {p1_4} (+/- {err_4})")
