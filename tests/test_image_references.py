import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# Mentions of image names that are not references to the repository's files
IGNORED_TEXT = ("testdata/",)

NAME = re.compile(r"(?<![\w+\-.])([\w+\-]+)\.(png|webp)\b")
# In HTML, only links count; example commands like "save image.png" do not
HTML_LINK = re.compile(r"""(?:src|href)\s*=\s*["']([^"']*)["']|url\(\s*["']?([^"')]*)""",
                       re.I)


def _git(*args):
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.split("\n")[:-1]


def _image_names(suffix):
    # Untracked images count, and deleted ones do not, so the tests
    # agree with the working tree before changes are staged
    return {os.path.basename(p)[:-len(suffix)]
            for p in _git("ls-files", "--cached", "--others",
                          "--exclude-standard", "*" + suffix)
            if (REPO / p).exists()}


def _references(suffix):
    refs = {}
    for path in _git("grep", "-lI", "--untracked", r"\." + suffix[1:]):
        if path.startswith(IGNORED_TEXT):
            continue
        text = (REPO / path).read_text(encoding="utf-8", errors="replace")
        if path.endswith((".html", ".htm")):
            text = "\n".join(m.group(1) or m.group(2) or ""
                             for m in HTML_LINK.finditer(text))
        for m in NAME.finditer(text):
            if "." + m.group(2) == suffix:
                refs.setdefault(m.group(1), set()).add(path)
    return refs


def test_no_references_to_converted_pngs():
    """Images converted to WebP are not still referenced as PNGs"""
    pngs = _image_names(".png")
    webps = _image_names(".webp")
    stale = {name: sorted(paths)
             for name, paths in _references(".png").items()
             if name in webps and name not in pngs}
    assert stale == {}


def test_webp_references_exist():
    """Every WebP image named in the sources is in the repository"""
    webps = _image_names(".webp")
    missing = {name: sorted(paths)
               for name, paths in _references(".webp").items()
               if name not in webps}
    assert missing == {}
