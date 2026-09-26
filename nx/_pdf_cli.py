"""CLI-binary PDF extraction (0.100.0): the subprocess path for PDF mentions,
mirroring how _duckdb.py wraps the DuckDB CLI instead of importing it.

Why: pymupdf4llm as a HARD dependency costs ~200 MB of venv (pymupdf 116 MB
plus numpy/onnxruntime/networkx dragged in) for a feature used only on PDF
mentions. The same extraction quality is reachable from small binaries on
PATH, tried in order:

1. ``mutool convert -F markdown``  (mupdf-tools, MuPDF >= 1.24) — markdown,
   closest to pymupdf4llm's output.
2. ``pdftotext -layout``           (poppler-utils, ubiquitous) — layouted
   plain text; no markdown structure but readable.
3. tesseract OCR                   — only when the text layer comes back
   empty (scanned pages): render pages with pdftoppm/mutool and OCR them.

The pymupdf4llm import in _pdf.py remains as the final fallback for
pip-only environments with none of these binaries; it is no longer a hard
dependency. Every entry point degrades to a note, never raises, and honors
NX_PDF_MAX_PAGES / NX_PDF_TIMEOUT exactly like the in-process path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ._env import _env_int
from ._pdf import _pdf_max_pages

# Wall-clock ceiling for one extraction (all pages, including OCR). A wedged
# CLI must not hold the turn; this mirrors _duckdb's contract.
_DEFAULT_TIMEOUT = 300

# OCR is far slower than a text-layer read; fewer pages by default so a
# 50-page scan doesn't stall the turn for minutes.
_OCR_MAX_PAGES = 10


def _timeout() -> int:
    return _env_int("NX_PDF_TIMEOUT", _DEFAULT_TIMEOUT)


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess | None:
    """subprocess.run with capture; None on missing binary/timeout/OS error."""
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout or None)
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None


def _page_count(path: str) -> int:
    """Page count via pdfinfo or mutool info; 0 when neither can say."""
    timeout = _timeout()
    r = _run(["pdfinfo", path], timeout)
    if r and r.returncode == 0:
        for line in (r.stdout or "").splitlines():
            if line.startswith("Pages:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    r = _run(["mutool", "info", path], timeout)
    if r and r.returncode == 0:
        for line in (r.stdout or "").splitlines():
            if "Pages" in line and ":" in line:
                tail = line.split(":", 1)[1].strip().split(" ")[0].strip("()")
                try:
                    return int(tail)
                except ValueError:
                    pass
    return 0


def _range(pages_n: int, cap: int) -> tuple[int, int]:
    """1-based (first, last) page range honoring the cap (cap<=0 = no cap)."""
    if cap > 0 and pages_n > cap:
        return 1, cap
    return 1, max(pages_n, 1)


def _mutool_markdown(path: str, first: int, last: int,
                     timeout: int) -> str | None:
    """mutool convert -F markdown (MuPDF >= 1.24); None when unavailable."""
    if not shutil.which("mutool"):
        return None
    with tempfile.TemporaryDirectory(prefix="nx-pdf-") as tmp:
        out = os.path.join(tmp, "doc.md")
        argv = ["mutool", "convert", "-F", "markdown", "-o", out,
                path, f"{first}-{last}"]
        r = _run(argv, timeout)
        if not r or r.returncode != 0:
            return None
        try:
            with open(out, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None


def _pdftotext(path: str, first: int, last: int, timeout: int) -> str | None:
    """pdftotext -layout to stdout; None when unavailable/failed."""
    if not shutil.which("pdftotext"):
        return None
    r = _run(["pdftotext", "-layout", "-f", str(first), "-l", str(last),
              path, "-"], timeout)
    if not r or r.returncode != 0:
        return None
    return r.stdout


def _ocr_pages(path: str, first: int, last: int,
               timeout: int) -> str | None:
    """OCR fallback for scanned PDFs: render pages to PNG, tesseract each.

    Requires pdftoppm (poppler) or mutool for rendering and tesseract for
    recognition. Returns None when either is missing or nothing readable
    came out — the caller keeps its "scanned images?" note.
    """
    if not shutil.which("tesseract"):
        return None
    last = min(last, first + _OCR_MAX_PAGES - 1)
    with tempfile.TemporaryDirectory(prefix="nx-pdf-ocr-") as tmp:
        prefix = os.path.join(tmp, "page")
        if shutil.which("pdftoppm"):
            r = _run(["pdftoppm", "-r", "150", "-png", "-f", str(first),
                      "-l", str(last), path, prefix], timeout)
            if not r or r.returncode != 0:
                return None
        elif shutil.which("mutool"):
            r = _run(["mutool", "draw", "-o", prefix + "-%d.png", "-r", "150",
                      path, f"{first}-{last}"], timeout)
            if not r or r.returncode != 0:
                return None
        else:
            return None
        try:
            pngs = sorted(n for n in os.listdir(tmp) if n.endswith(".png"))
        except OSError:
            return None
        chunks: list[str] = []
        for png in pngs:
            r = _run(["tesseract", os.path.join(tmp, png), "stdout"],
                     timeout)
            if r and r.returncode == 0 and (r.stdout or "").strip():
                chunks.append(r.stdout.strip())
        return "\n\n".join(chunks) if chunks else None


def extract_pdf_cli(path: str) -> str | None:
    """Text/markdown for *path* via CLI binaries, or None when no binary
    chain is available (the caller falls back to the in-process extractor).

    Never raises; returns "" only when a toolchain ran but produced no text
    (scanned PDF without tesseract), which the caller surfaces as its note.
    """
    timeout = _timeout()
    pages_n = _page_count(path)
    first, last = _range(pages_n, _pdf_max_pages())
    md = _mutool_markdown(path, first, last, timeout)
    if md is None:
        md = _pdftotext(path, first, last, timeout)
    if md is not None and md.strip():
        return md
    # Empty text layer → try OCR (also covers a toolchain that produced an
    # empty-but-successful result on a scanned document).
    ocr = _ocr_pages(path, first, last, timeout)
    if ocr and ocr.strip():
        return ocr
    if md is None and pages_n == 0 and not shutil.which("pdftotext") \
            and not shutil.which("mutool"):
        return None  # no toolchain at all → caller tries pymupdf4llm
    return md or ""
