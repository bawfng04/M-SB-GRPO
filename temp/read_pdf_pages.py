import pypdf, sys
sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader(r"d:\Projects\M-SB-GRPO\paper_final\SB-GRPO.pdf")
num_pages = len(reader.pages)
print(f"Total pages: {num_pages}")

for idx, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    first_few = " | ".join(lines[:3]) if lines else "EMPTY"
    last_few = " | ".join(lines[-2:]) if lines else "EMPTY"
    print(f"\n--- Page {idx} ({len(lines)} lines) ---")
    print(f"Start: {first_few[:120]}")
    print(f"End:   {last_few[:120]}")
