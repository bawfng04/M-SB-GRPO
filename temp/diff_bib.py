import difflib

with open(r'd:\Projects\M-SB-GRPO\paper\references.bib', 'r', encoding='utf-8', errors='ignore') as f1:
    lines1 = f1.readlines()
with open(r'd:\Projects\M-SB-GRPO\paper_final\references.bib', 'r', encoding='utf-8', errors='ignore') as f2:
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines2, lines1, fromfile='paper_final/references.bib', tofile='paper/references.bib'))
print(f"Diff lines: {len(diff)}")
for l in diff[:40]:
    print(l, end='')
