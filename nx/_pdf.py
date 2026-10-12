"""PDF `@mention` support: turn a mentioned `.pdf` into markdown text so it
reaches the model the same way a text-file mention does.

Extraction prefers small CLI binaries (see _pdf_cli.py: `mutool convert
-F markdown`, then `pdftotext -layout`, then tesseract OCR for scanned
pages) — the same cascade idea as the duckdb tool, which shells out to the
DuckDB CLI rather than importing it. Only when none of those binaries is on
PATH does it fall back to pymupdf4llm, which is NO LONGER a hard pyproject
dependency (0.100.0 demoted it: ~200 MB of venv for a lazily-used feature)
but stays lazily imported so pip-only installs keep working. Pure function,
no globals/state.
"""
from __future__ import annotations

from ._env import _env_int

# Pages extracted per mention. A 300-page PDF inlined whole would blow any
# context window, and extraction gets slow on long documents; the mention
# header names the range it actually carried so the truncation is visible.
PDF_MAX_PAGES = 50

# Install hint for the no-toolchain case (soft note, never an abort).
_INSTALL_HINT = ("PDF extraction needs mupdf-tools or poppler-utils on PATH "
                 "(or `uv pip install pymupdf4llm`)")


def _pdf_max_pages() -> int:
    """Env read at CALL time (NX_* resolved at import would freeze the shell
    value at collection — the CONTRIBUTING rule)."""
    return _env_int("NX_PDF_MAX_PAGES", PDF_MAX_PAGES)


def _header(n_pages: int) -> str:
    """Range header for an extraction whose page count is known."""
    cap = _pdf_max_pages()
    shown = min(n_pages, cap) if (cap > 0 and n_pages > cap) else n_pages
    if shown < n_pages:
        return (f"(PDF, {n_pages} pages — carried the first {shown}; "
                f"raise NX_PDF_MAX_PAGES for more)\n")
    return f"(PDF, {n_pages} page{'s' if n_pages != 1 else ''})\n"


def _tail(head: str, path: str, md: str) -> str:
    """Shared result folding for the CLI path: empty → scanned note, else
    count pages via the pdfinfo/mutool probe for the same framing the
    in-process path produces."""
    md = md.strip()
    if not md:
        return head + f"[{path}: PDF carried no extractable text (scanned images?)]"
    from ._pdf_cli import _page_count  # lazy: sibling module, no cycle
    n_pages = _page_count(path)
    header = _header(n_pages) if n_pages > 0 else ""
    return head + header + md + "\n"


def extract_pdf(path: str, token: str | None = None) -> str:
    """Return a `--- @token ---` block holding the PDF's text as markdown, or
    a one-line note when extraction isn't possible (missing dependency, open
    failure, empty). Never raises: a mention must degrade to text, not abort
    the turn."""
    label = token or path
    head = f"\n\n--- @{label} ---\n"
    # 1) CLI binaries (mutool convert → pdftotext → tesseract OCR). This is
    #    the primary path: no in-process heavyweight deps.
    try:
        from ._pdf_cli import extract_pdf_cli
        cli_text = extract_pdf_cli(path)
    except Exception as e:  # noqa: BLE001 — the CLI chain must never abort a turn
        return head + f"ERROR: PDF extraction failed: {type(e).__name__}: {e}"
    if cli_text is not None:
        return _tail(head, path, cli_text)
    # 2) No toolchain on PATH → in-process pymupdf4llm fallback (lazy).
    try:
        import pymupdf4llm  # lazy: heavy, demoted from hard dep in 0.100.0  # pyrefly: ignore[missing-import]  # optional dep, guarded by this try/except ImportError
    except ImportError:
        return head + f"(a PDF you asked about — not inlined: {_INSTALL_HINT})"
    # pymupdf4llm depends on PyMuPDF, so `pymupdf` is importable alongside it;
    # we open the doc ourselves to learn the page count and cap the extraction.
    try:
        import pymupdf  # pyrefly: ignore[missing-import]  # optional dep (pymupdf4llm's own dep); ImportError caught below
        doc = pymupdf.open(path)
    except Exception as e:  # noqa: BLE001 — any failure becomes a note
        return head + f"ERROR: could not open PDF: {type(e).__name__}: {e}"
    try:
        n_pages = doc.page_count
        if n_pages == 0:
            return head + f"[{path}: PDF has no pages]"
        cap = _pdf_max_pages()
        if cap > 0 and n_pages > cap:
            pages: list[int] | None = list(range(cap))
        else:
            pages = None  # whole document (also when cap<=0 = no cap)
        try:
            md = pymupdf4llm.to_markdown(doc, pages=pages)
        except Exception as e:  # noqa: BLE001
            return head + f"ERROR: PDF extraction failed: {type(e).__name__}: {e}"
    finally:
        doc.close()
    md = md.strip()
    if not md:
        return head + f"[{path}: PDF carried no extractable text (scanned images?)]"
    return head + _header(n_pages) + md
