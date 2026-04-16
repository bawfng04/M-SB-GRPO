import pymupdf4llm
from pathlib import Path

# convert all paper(.pdf) to markdown(.md) in the current directory,
# and save to papers-txt/ directory


def main():
    source_dir = Path(".")
    output_dir = source_dir / "papers-txt"
    output_dir.mkdir(exist_ok=True)
    pdf_files = list(source_dir.glob("*.pdf"))
    if not pdf_files:
        print("pdf not found")
        return
    print(f"scan {len(pdf_files)} file...")
    for pdf in pdf_files:
        print(f"processing {pdf.name}...")
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf))
            out_file = output_dir / f"{pdf.stem}.md"
            out_file.write_text(md_text, encoding="utf-8")
            print(f" -> {out_file.name}")
        except Exception as e:
            print(f"error at {pdf.name}: {e}")
    print("\nDone!")


if __name__ == "__main__":
    main()
