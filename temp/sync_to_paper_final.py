import shutil
from pathlib import Path

src_base = Path("paper")
dst_base = Path("paper_final")

files_to_sync = [
    "SB-GRPO.tex",
    "references.bib",
    "section/abstract.tex",
    "section/introduce.tex",
    "section/related-work.tex",
    "section/methodoloy.tex",
    "section/dataset-environment.tex",
    "section/experiment-result.tex",
    "section/conclusion.tex"
]

for rel in files_to_sync:
    src = src_base / rel
    dst = dst_base / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Synced {rel}: {src.stat().st_size} bytes -> {dst.stat().st_size} bytes")

print("Sync completed successfully.")
