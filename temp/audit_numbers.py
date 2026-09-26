import re
from pathlib import Path

tex_dir = Path("paper/section")
for tf in sorted(tex_dir.glob("*.tex")):
    content = tf.read_text(encoding="utf-8")
    # find percentages, plus/minus, and numbers
    matches = re.findall(r'(\+?\-?\d+\.?\d*\%|\b\d+\.?\d*\s*\\pm\s*\d+\.?\d*|\b\d+\.?\d*\b)', content)
    print(f"\n=== {tf.name} ===")
    unique_nums = sorted(set(m for m in matches if any(c.isdigit() for c in m)))
    print(", ".join(unique_nums[:30]))
