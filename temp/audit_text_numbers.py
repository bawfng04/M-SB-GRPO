import re
from pathlib import Path

content = Path("paper/section/experiment-result.tex").read_text(encoding="utf-8")

# Let's extract lines that have numbers or percentages outside of tables
lines = content.splitlines()
in_table = False
for idx, line in enumerate(lines, 1):
    if "\\begin{table}" in line:
        in_table = True
    elif "\\end{table}" in line:
        in_table = False
        continue
    if not in_table:
        # find numbers
        nums = re.findall(r'(\d+[\.,]?\d*\%?)', line)
        if nums and not line.strip().startswith("%") and not line.strip().startswith("\\caption") and not line.strip().startswith("\\label"):
            print(f"L{idx}: {line.strip()}")
