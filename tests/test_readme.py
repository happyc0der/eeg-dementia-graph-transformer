"""README consistency: generated tables match results/, links resolve, no local absolute paths."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")  # C:\... or C:/..., but not https://


def _generator():
    spec = importlib.util.spec_from_file_location("make_readme_tables", ROOT / "scripts" / "make_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_tables_match_results():
    gen = _generator()
    text = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert gen.render(text) == text, "README.md is out of date: run `uv run python scripts/make_readme_tables.py`"


def test_drive_path_pattern():
    assert DRIVE_PATH.search(r"see C:\AI\x") and DRIVE_PATH.search("(C:/AI)")
    assert not DRIVE_PATH.search("https://example.org") and not DRIVE_PATH.search("AD:CN")


@pytest.mark.parametrize("readme", ["README.md", "legacy/README.md"])
def test_readme_links_resolve_and_no_local_paths(readme):
    path = ROOT / readme
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
    local = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]
    assert local, "no local links found"
    missing = [t for t in local if not (path.parent / t).exists()]
    assert not missing, f"broken relative links in {readme}: {missing}"
    m = DRIVE_PATH.search(text)
    assert m is None, f"absolute local path in {readme}: {text[max(0, m.start() - 20): m.end() + 20]!r}"
