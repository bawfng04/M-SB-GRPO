import pypdf, sys
sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader(r"d:\Projects\M-SB-GRPO\paper_final\SB-GRPO.pdf")

for p in [9, 10, 11]:
    text = reader.pages[p-1].extract_text()
    print(f"\n================ PAGE {p} ================")
    print(text)
