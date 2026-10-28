import pytest


class _MemoryHistory:
    """Stand-in for ObjectHistory that keeps saved history files in memory by tag."""

    saved = {}

    def __init__(self, tag, unversioned=True):
        self._tag = tag

    def load(self):
        return _MemoryHistory.saved.get(self._tag)

    def save(self, obj):
        import json

        # Round trip through JSON like the real history file.
        _MemoryHistory.saved[self._tag] = json.loads(json.dumps(obj))

    def backup(self):
        return None


@pytest.fixture
def history(test_production_session, monkeypatch):
    import chimerax.core.history

    monkeypatch.setattr(chimerax.core.history, "ObjectHistory", _MemoryHistory)
    _MemoryHistory.saved = {}
    from chimerax.core.filehistory import FileHistory

    yield FileHistory(test_production_session)
    _MemoryHistory.saved = {}


def _add_files(h, *paths):
    import time

    specs = []
    for path in paths:
        h.remember_file(path, "pdb", [])
        spec = h.find_file(path)
        # Guarantee distinct access times for ordering.
        spec.access_time = time.time() + len(specs)
        specs.append(spec)
    return specs


def _paths(specs):
    import os.path

    return [os.path.basename(f.path) for f in specs]


def test_unsorted_files_most_recent_first(history):
    _add_files(history, "/tmp/a.pdb", "/tmp/b.pdb", "/tmp/c.pdb")
    assert history.sections == []
    assert _paths(history.section_files(None)) == ["c.pdb", "b.pdb", "a.pdb"]


def test_move_file_into_section_keeps_position_on_reopen(history):
    a, b, c = _add_files(history, "/tmp/a.pdb", "/tmp/b.pdb", "/tmp/c.pdb")
    history.add_section("Project X")
    history.move_file(a, "Project X")
    history.move_file(c, "Project X")
    assert _paths(history.section_files("Project X")) == ["a.pdb", "c.pdb"]
    assert _paths(history.section_files(None)) == ["b.pdb"]
    assert history.file_section(a) == "Project X"

    # Reopening a file in a section does not reorder the section.
    history.remember_file("/tmp/a.pdb", "pdb", [])
    assert _paths(history.section_files("Project X")) == ["a.pdb", "c.pdb"]


def test_move_file_before_another(history):
    a, b, c = _add_files(history, "/tmp/a.pdb", "/tmp/b.pdb", "/tmp/c.pdb")
    history.add_section("P")
    for f in (a, b):
        history.move_file(f, "P")
    history.move_file(c, "P", before=a)
    assert _paths(history.section_files("P")) == ["c.pdb", "a.pdb", "b.pdb"]
    # Moving within the same section reorders it.
    history.move_file(c, "P")
    assert _paths(history.section_files("P")) == ["a.pdb", "b.pdb", "c.pdb"]
    # Moving to None returns the file to the unsorted files.
    history.move_file(b, None)
    assert _paths(history.section_files("P")) == ["a.pdb", "c.pdb"]
    assert "b.pdb" in _paths(history.section_files(None))


def test_section_names_validated(history):
    history.add_section("P")
    with pytest.raises(ValueError):
        history.add_section("P")
    with pytest.raises(ValueError):
        history.add_section("   ")
    assert history.add_section("  Q  ") == "Q"
    with pytest.raises(ValueError):
        history.rename_section("P", "Q")


def test_rename_move_and_remove_section(history):
    (a,) = _add_files(history, "/tmp/a.pdb")
    for name in ("P", "Q", "R"):
        history.add_section(name)
    history.move_file(a, "Q")
    history.current_section = "Q"
    history.rename_section("Q", "Q2")
    assert history.sections == ["P", "Q2", "R"]
    assert history.current_section == "Q2"
    assert history.file_section(a) == "Q2"

    history.move_section("R", 0)
    assert history.sections == ["R", "P", "Q2"]

    history.remove_section("Q2")
    assert history.sections == ["R", "P"]
    assert history.current_section is None
    assert history.file_section(a) is None
    assert _paths(history.section_files(None)) == ["a.pdb"]


def test_new_files_added_to_current_section(history):
    history.add_section("P")
    history.current_section = "P"
    _add_files(history, "/tmp/a.pdb")
    assert _paths(history.section_files("P")) == ["a.pdb"]
    with pytest.raises(ValueError):
        history.current_section = "No such section"


def test_remove_file(history):
    a, b = _add_files(history, "/tmp/a.pdb", "/tmp/b.pdb")
    history.add_section("P")
    history.move_file(a, "P")
    history.remove_file(a)
    assert history.find_file("/tmp/a.pdb") is None
    assert history.section_files("P") == []
    assert _paths(history.section_files(None)) == ["b.pdb"]


