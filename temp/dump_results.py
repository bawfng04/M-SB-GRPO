import json, os, glob
from pathlib import Path

results_dir = Path(r"d:\Projects\M-SB-GRPO\compare\open-r1\results")

print("=== SCANNING JSON FILES IN compare/open-r1/results ===")
json_files = sorted(results_dir.glob("**/*.json"))
for jf in json_files:
    rel = jf.relative_to(results_dir)
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Check structure
            results = data.get("results", {})
            print(f"\n--- File: {rel} ---")
            for bench, metrics in results.items():
                print(f"  Benchmark: {bench}")
                for k, v in metrics.items():
                    print(f"    {k}: {v}")
    except Exception as e:
        print(f"Error reading {rel}: {e}")
