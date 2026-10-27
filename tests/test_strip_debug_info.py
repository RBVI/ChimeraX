import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest


def _load_clean_app_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "utils" / "clean_app.py"
    spec = importlib.util.spec_from_file_location("clean_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _elf():
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2   # 64-bit
    header[5] = 1   # little-endian
    header[18:20] = (62).to_bytes(2, "little")
    return bytes(header)


def test_elf_files_skips_links_and_other_files(tmp_path):
    clean_app = _load_clean_app_module()
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.so").write_bytes(_elf())
    os.link(lib / "a.so", lib / "a.so.1")
    (lib / "a.so.2").symlink_to("a.so")
    (lib / "b.dylib").write_bytes(b"\xcf\xfa\xed\xfe" + bytes(60))
    (lib / "README").write_text("not a binary")
    # one group of hard links, and no symlinks or other files
    found = clean_app.elf_files(str(tmp_path))
    assert len(found) == 1
    assert sorted(os.path.basename(p) for p in found[0]) == ["a.so", "a.so.1"]


def _sections(path):
    return subprocess.run(["readelf", "-S", "-W", str(path)],
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(
    sys.platform != "linux"
    or not all(shutil.which(t) for t in ("cc", "strip", "readelf")),
    reason="needs a Linux toolchain")
def test_strip_debug_info(tmp_path):
    clean_app = _load_clean_app_module()
    source = tmp_path / "f.c"
    source.write_text("int chimerax_f(int x) { return x * 2; }\n")
    lib = tmp_path / "lib" / "libf.so"
    lib.parent.mkdir()
    subprocess.run(["cc", "-g", "-shared", "-fPIC", "-o", str(lib),
                    str(source)], check=True)
    (tmp_path / "README").write_text("not a binary")
    assert ".debug_info" in _sections(lib)

    freed = clean_app.strip_debug_info(str(tmp_path))

    assert freed > 0
    sections = _sections(lib)
    assert ".debug_info" not in sections
    # symbols stay, so backtraces still have function names
    assert ".symtab" in sections
    assert (tmp_path / "README").read_text() == "not a binary"


@pytest.mark.skipif(
    sys.platform != "linux"
    or not all(shutil.which(t) for t in ("cc", "strip", "readelf")),
    reason="needs a Linux toolchain")
def test_strip_debug_info_leaves_mapped_file_alone(tmp_path):
    """The app's Python has its libraries mapped while clean_app runs, so
    they must be replaced rather than rewritten in place"""
    clean_app = _load_clean_app_module()
    source = tmp_path / "f.c"
    source.write_text("int chimerax_f(int x) { return x * 2; }\n")
    lib = tmp_path / "lib" / "libf.so"
    lib.parent.mkdir()
    subprocess.run(["cc", "-g", "-shared", "-fPIC", "-o", str(lib),
                    str(source)], check=True)
    os.link(lib, tmp_path / "lib" / "libf.so.1")
    original = lib.read_bytes()
    # an open file stands in for a mapping of the original inode
    with open(lib, "rb") as mapped:
        clean_app.strip_debug_info(str(tmp_path))
        assert mapped.read() == original
    assert ".debug_info" not in _sections(lib)
    # the hard link points at the stripped file
    assert os.path.samefile(lib, tmp_path / "lib" / "libf.so.1")
    assert not list(tmp_path.rglob("*" + clean_app.STRIP_SUFFIX))
