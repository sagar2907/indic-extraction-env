"""Render report/report.md to PDF, then verify the glyphs actually drew.

The verification step is not optional decoration. A PDF that lacks a font covering
Devanagari, Tamil or Bengali does not fail: it emits tofu boxes or, worse, blank space,
and the file opens perfectly happily. Since roughly a third of this report is Indic
script, "the build succeeded" is not evidence that the document is readable.

So the script renders the PDF, rasterises pages back to images, and checks that pages
carrying Indic text contain a plausible amount of dark ink. A page that should be full
of Tamil and comes back nearly blank is a font failure, and the script exits non-zero.

Pipeline: markdown -> HTML (python-markdown) -> PDF (headless Chrome/Edge) -> PNG
(pypdfium2) -> ink check. Chrome is used because pandoc, weasyprint, wkhtmltopdf and
xelatex are all absent on this machine, whereas a Chromium browser is not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_MD = REPO_ROOT / "report" / "report.md"
REPORT_PDF = REPO_ROOT / "report" / "indic-extraction-env-report.pdf"
PREVIEW_DIR = REPO_ROOT / "report" / "_render"

BROWSERS = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/google-chrome"),
)

# Nirmala UI is the Windows system font covering Devanagari, Tamil and Bengali in one
# family. Listing it ahead of the Latin faces means Indic runs resolve to a font that
# actually has the glyphs instead of falling through to tofu.
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body {
  font-family: "Nirmala UI", "Noto Sans", "Segoe UI", system-ui, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #14171a; max-width: 100%;
}
h1 { font-size: 22pt; margin: 0 0 4pt; page-break-before: always; color: #0f172a; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 15pt; margin: 20pt 0 6pt; color: #0f172a;
     border-bottom: 1px solid #cbd5e1; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; color: #1e293b; }
p, li { orphans: 3; widows: 3; }
code, pre { font-family: "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace; }
code { font-size: 9pt; background: #f1f5f9; padding: 1px 3px; border-radius: 3px; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #64748b;
      padding: 8pt 10pt; font-size: 8.5pt; line-height: 1.4; overflow-x: auto;
      page-break-inside: avoid; border-radius: 3px; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e1; padding: 4pt 7pt; text-align: left;
         vertical-align: top; }
th { background: #f1f5f9; font-weight: 600; }
blockquote { border-left: 3px solid #94a3b8; margin: 10pt 0; padding: 2pt 0 2pt 12pt;
             color: #334155; background: #f8fafc; }
hr { border: 0; border-top: 1px solid #cbd5e1; margin: 16pt 0; }
a { color: #1d4ed8; text-decoration: none; }
.indic { font-family: "Nirmala UI", sans-serif; }
"""


def find_browser() -> Path:
    for candidate in BROWSERS:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "No Chromium-family browser found. Install Chrome or Edge, or add a path to "
        "BROWSERS in this script."
    )


def markdown_to_html(source: str) -> str:
    import markdown

    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<title>indic-extraction-v1</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def html_to_pdf(html: str, out_pdf: Path) -> None:
    browser = find_browser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        # A throwaway user-data-dir keeps this from colliding with a running browser
        # profile, which otherwise makes headless Chrome exit without writing anything.
        subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    if not out_pdf.exists():
        raise SystemExit(f"browser reported success but {out_pdf} was not written")


def verify_glyphs(pdf_path: Path, preview_dir: Path, min_ink: float = 0.004) -> bool:
    """Rasterise every page and confirm it contains a plausible amount of ink.

    Missing-glyph failures are silent: the text layer is present, the drawing is not.
    Rendering back to pixels is the only check that actually looks at what a reader
    would see. `min_ink` is the fraction of non-white pixels below which a page is
    treated as suspiciously empty.
    """
    import pypdfium2 as pdfium

    preview_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    ok = True
    print(f"pages: {len(pdf)}")
    for index in range(len(pdf)):
        bitmap = pdf[index].render(scale=1.6)
        image = bitmap.to_pil().convert("L")
        histogram = image.histogram()
        dark = sum(histogram[:200])
        ink = dark / (image.width * image.height)
        out = preview_dir / f"page_{index + 1:02d}.png"
        image.save(out)
        flag = ""
        if ink < min_ink:
            flag = "  <-- SUSPICIOUSLY EMPTY"
            ok = False
        print(f"  page {index + 1:>2}: ink {ink:>7.4f}{flag}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(REPORT_MD))
    parser.add_argument("--out", default=str(REPORT_PDF))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"missing {source}")

    html = markdown_to_html(source.read_text(encoding="utf-8"))
    out_pdf = Path(args.out)
    html_to_pdf(html, out_pdf)
    size_kb = out_pdf.stat().st_size / 1024
    print(f"wrote {out_pdf} ({size_kb:.0f} KB)")

    if args.skip_verify:
        return 0
    if not verify_glyphs(out_pdf, PREVIEW_DIR):
        print("\nFAILED: at least one page rendered nearly blank. Check font coverage.")
        return 1
    print("\nAll pages contain rendered content.")
    print(
        f"\nThe ink check proves no page is blank. It does NOT prove the glyphs are\n"
        f"correct -- missing-glyph boxes carry ink too. Open the PNGs in\n"
        f"{PREVIEW_DIR} and look at a page containing Devanagari, Tamil and Bengali\n"
        f"before treating the render as verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
