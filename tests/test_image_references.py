import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# Mentions of image names that are not references to the repository's files:
# test data, this file's made-up names, and saved web pages (whose resources
# are archived under their original URLs)
IGNORED_TEXT = ("testdata/", "tests/test_image_references.py")
IGNORED_SUFFIXES = (".webarchive",)

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
        if path.startswith(IGNORED_TEXT) or path.endswith(IGNORED_SUFFIXES):
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


def _load_ui_icons(directory):
    """Load chimerax.ui.icons from a copy in directory, so get_icon_path()
    looks for icons there"""
    import importlib.util
    source = REPO / "src" / "bundles" / "ui" / "src" / "icons" / "__init__.py"
    path = directory / "__init__.py"
    path.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("ui_icons", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_icon_path_finds_any_format(tmp_path):
    icons = _load_ui_icons(tmp_path)
    for fn in ("new.png", "both.png", "both.webp", "vector.svg"):
        (tmp_path / fn).write_bytes(b"")
    assert icons.get_icon_path("new") == str(tmp_path / "new.png")
    assert icons.get_icon_path("both") == str(tmp_path / "both.webp")
    assert icons.get_icon_path("vector") == str(tmp_path / "vector.svg")
    assert icons.get_icon_path("missing") == str(tmp_path / "missing.webp")


def test_ui_icons_found():
    """Each icon in chimerax.ui.icons is found by name"""
    icon_dir = REPO / "src" / "bundles" / "ui" / "src" / "icons"
    import importlib.util
    spec = importlib.util.spec_from_file_location("ui_icons",
                                                  icon_dir / "__init__.py")
    icons = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(icons)
    for path in icon_dir.iterdir():
        if path.suffix in icons.ICON_SUFFIXES:
            assert os.path.exists(icons.get_icon_path(path.stem))
