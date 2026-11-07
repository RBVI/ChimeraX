"""utils/make_startup_prefetch_list.py turns a recorded prefetch list into
one shipped with the application, relative to the application directory."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "make_startup_prefetch_list",
    os.path.join(os.path.dirname(__file__), "..", "utils", "make_startup_prefetch_list.py"))
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_entries_become_relative_and_outside_files_are_dropped(tmp_path):
    app = tmp_path / "ChimeraX.app"
    lib = os.path.join(str(app), "lib", "libQt6Core.so.6")
    pyc = os.path.join(str(app), "lib", "mod.cpython-314.pyc")
    user = os.path.join(str(tmp_path), "cache", "toolshed.cache")
    recorded = f"{lib}\t0+4096,8192+4096\n{pyc}\n{user}\n"
    shipped = tool.relative_list(recorded, str(app))
    assert shipped == (os.path.join("lib", "libQt6Core.so.6") + "\t0+4096,8192+4096\n"
                       + os.path.join("lib", "mod.cpython-314.pyc") + "\n")


def test_a_sibling_directory_with_the_same_prefix_is_not_inside(tmp_path):
    app = tmp_path / "ChimeraX.app"
    sibling = os.path.join(str(tmp_path), "ChimeraX.app-old", "lib", "x.so")
    assert tool.relative_list(sibling + "\n", str(app)) == ""
