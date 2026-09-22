"""PDF `@mention` support: turn a mentioned `.pdf` into markdown text so it
reaches the model the same way a text-file mention does.

Extraction goes through pymupdf4llm (PyMuPDF's LLM-oriented markdown
extractor), which keeps headings, lists, and tables as markdown instead of
the flat word soup a naive text dump produces. The dependency is OPTIONAL and
heavy, so the import is lazy — `nx` starts fine without it, and a mention of a
PDF when it is missing carries a one-line note naming the install command
(mirroring how an image mention with NX_VISION off names its gate), not an
error. Pure function, no globals/state.
"""
from __future__ import annotations

from ._env import _env_int

# Pages extracted per mention. A 300-page PDF inlined whole would blow any
# context window, and pymupdf4llm gets slow on long documents; the mention
# header names the range it actually carried so the truncation is visible.
PDF_MAX_PAGES = 50

# Named dependency, not a hard one: lighter installs that never mention a PDF
# shouldn't pay for PyMuPDF's ~30 MB.
_INSTALL_HINT = ("pymupdf4llm is not installed — PDF mentions need it: "
                 "`uv pip install pymupdf4llm` (or `pip install 'nx[pdf]'`)")


def _pdf_max_pages() -> int:
    """Env read at CALL time (NX_* resolved at import would freeze the shell
    value at collection — the CONTRIBUTING rule)."""
    return _env_int("NX_PDF_MAX_PAGES", PDF_MAX_PAGES)


def extract_pdf(path: str, token: str | None = None) -> str:
    """Return a `--- @token ---` block holding the PDF's text as markdown, or
    a one-line note when extraction isn't possible (missing dependency, open
    failure, empty). Never raises: a mention must degrade to text, not abort
    the turn."""
    label = token or path
    head = f"\n\n--- @{label} ---\n"
    try:
        import pymupdf4llm  # lazy: optional, heavy
    except ImportError:
        return head + f"(a PDF you asked about — not inlined: {_INSTALL_HINT})"
    # pymupdf4llm depends on PyMuPDF, so `pymupdf` is importable alongside it;
    # we open the doc ourselves to learn the page count and cap the extraction.
    try:
        import pymupdf
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
    shown = min(n_pages, cap) if (cap > 0 and n_pages > cap) else n_pages
    if shown < n_pages:
        header = (f"(PDF, {n_pages} pages — carried the first {shown}; "
                  f"raise NX_PDF_MAX_PAGES for more)\n")
    else:
        header = f"(PDF, {n_pages} page{'s' if n_pages != 1 else ''})\n"
    return head + header + md
