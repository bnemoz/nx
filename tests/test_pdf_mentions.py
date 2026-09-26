"""0.65.0: @pdf mentions — an @doc.pdf token is extracted to markdown text
via the extraction chain and inlined like any other mention. Since 0.100.0
the primary path is CLI binaries (mutool convert -F markdown → pdftotext →
tesseract OCR, see nx/_pdf_cli.py); pymupdf4llm is a lazily-imported fallback
(demoted from a hard dependency in 0.100.0). The fallback tests drive the
in-process path through fake modules injected on sys.modules (offline, no
PyMuPDF needed); CLI-path tests fake the binaries with stub scripts.
"""
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_VISION", "NX_PDF_MAX_PAGES"):
    os.environ.pop(k, None)

import pytest


def _no_toolchain(monkeypatch):
    """Force the in-process fallback path: hide the CLI binaries the CLI
    extractor looks for (a bare test env usually has none of them anyway,
    but poppler-utils is common on CI images)."""
    import shutil
    orig_which = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda n: None if n in ("mutool", "pdftotext",
                                                "pdfinfo", "pdftoppm",
                                                "tesseract") else orig_which(n))


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    os.environ.pop("NX_PDF_MAX_PAGES", None)


def _pdf(tmp, name="doc.pdf", body=b"%PDF-1.4 fake"):
    p = tmp / name
    p.write_bytes(body)
    return p


class _FakeDoc:
    def __init__(self, n_pages):
        self.page_count = n_pages
        self.closed = False

    def close(self):
        self.closed = True


def _install_fake(monkeypatch, *, pages=3, markdown="# Title\n\nBody text.\n",
                  n_pages=None, open_raises=None, md_raises=None):
    """Inject fake `pymupdf` / `pymupdf4llm` modules. Returns a dict recording
    the `pages=` argument to_markdown received, so the cap can be asserted."""
    seen = {}
    fake_pt = types.ModuleType("pymupdf")
    n = n_pages if n_pages is not None else pages

    def _open(path):
        if open_raises is not None:
            raise open_raises
        return _FakeDoc(n)

    fake_pt.open = _open
    fake_md = types.ModuleType("pymupdf4llm")

    def _to_markdown(doc, pages=None):
        seen["pages"] = pages
        if md_raises is not None:
            raise md_raises
        return markdown

    fake_md.to_markdown = _to_markdown
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pt)
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake_md)
    return seen


def test_pdf_mention_inlines_markdown(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    seen = _install_fake(monkeypatch, markdown="# Heading\n\nparagraph\n")
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("summarize @doc.pdf")
    assert isinstance(out, str)
    assert "--- @doc.pdf ---" in out
    assert "# Heading" in out and "paragraph" in out
    assert "PDF, 3 pages" in out
    assert seen["pages"] is None  # 3 pages under the 50 default → whole doc


def test_pdf_mention_missing_dependency_is_soft_note(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    # pymupdf4llm was demoted from a hard dependency in 0.100.0 (CLI binaries
    # are the primary path), but the in-process fallback import stays lazy: a
    # missing toolchain AND missing pymupdf4llm still degrade to a one-line
    # note naming the fix, never an aborted turn.
    monkeypatch.setitem(sys.modules, "pymupdf4llm", None)
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("read @doc.pdf")
    assert isinstance(out, str)
    assert "not inlined" in out and "pymupdf4llm" in out


def test_pdf_mention_cli_toolchain_used(nx_fresh, monkeypatch, tmp_path):
    """0.100.0: when mutool/pdftotext are on PATH the CLI chain runs and the
    in-process pymupdf import is never touched."""
    nx, tmp = nx_fresh
    _pdf(tmp)
    # Fake binaries: a stub 'pdftotext' and 'pdfinfo' on a temp PATH.
    bindir = tmp / "bin"
    bindir.mkdir()
    (bindir / "pdfinfo").write_text(
        "#!/bin/sh\necho 'Pages:          3'\n")
    (bindir / "pdftotext").write_text(
        "#!/bin/sh\n# -layout -f N -l N file.pdf -\necho 'Layouted text from CLI'\n")
    for p in (bindir / "pdfinfo", bindir / "pdftotext"):
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    import nx._pdf as pdf_mod
    assert pdf_mod.extract_pdf(str(tmp / "doc.pdf")) \
        .endswith("(PDF, 3 pages)\nLayouted text from CLI\n") or \
        "Layouted text from CLI" in pdf_mod.extract_pdf(str(tmp / "doc.pdf"))


def test_pdf_page_cap_truncates_and_names_range(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    seen = _install_fake(monkeypatch, n_pages=120)
    os.environ["NX_PDF_MAX_PAGES"] = "10"
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("@doc.pdf")
    assert seen["pages"] == list(range(10))
    assert "120 pages" in out and "first 10" in out


def test_pdf_cap_zero_means_no_cap(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    seen = _install_fake(monkeypatch, n_pages=120)
    os.environ["NX_PDF_MAX_PAGES"] = "0"
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("@doc.pdf")
    assert seen["pages"] is None
    assert "120 pages" in out and "first" not in out


def test_open_failure_degrades_to_error(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    _install_fake(monkeypatch, open_raises=RuntimeError("corrupt"))
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("@doc.pdf")
    assert isinstance(out, str) and "ERROR: could not open PDF" in out


def test_empty_extraction_note(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    _pdf(tmp)
    _no_toolchain(monkeypatch)
    _install_fake(monkeypatch, markdown="   \n")
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("@doc.pdf")
    assert "no extractable text" in out


def test_non_pdf_mention_untouched(nx_fresh):
    nx, tmp = nx_fresh
    (tmp / "a.txt").write_text("plain text\n")
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("see @a.txt")
    assert "plain text" in out and "PDF" not in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
