import filecmp, os
from pathlib import Path

p_paper = Path("paper")
p_final = Path("paper_final")

all_paper_files = set(p.relative_to(p_paper).as_posix() for p in p_paper.rglob("*") if p.is_file())
all_final_files = set(p.relative_to(p_final).as_posix() for p in p_final.rglob("*") if p.is_file())

print("Only in paper:", all_paper_files - all_final_files)
print("Only in paper_final:", all_final_files - all_paper_files)

common = sorted(all_paper_files & all_final_files)
print("\nDifferences in common files:")
for rel in common:
    f1 = p_paper / rel
    f2 = p_final / rel
    if not filecmp.cmp(f1, f2, shallow=False):
        print(f"DIFF: {rel} (paper: {f1.stat().st_size}b, paper_final: {f2.stat().st_size}b)")
    else:
        print(f"SAME: {rel}")
