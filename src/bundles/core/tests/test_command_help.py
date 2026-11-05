import pytest

from chimerax.core import toolshed
from chimerax.core.commands import cli


@pytest.mark.parametrize(
    ("words", "suffix"),
    [(["a&b"], ""), (["a&b", "show"], "#show"), (["~a&b"], "#~a&amp;b")],
)
def test_help_url_escapes_command_name(tmp_path, monkeypatch, words, suffix):
    command_dir = tmp_path / "user" / "commands"
    command_dir.mkdir(parents=True)
    (command_dir / "a&b.html").touch()
    monkeypatch.setattr(toolshed, "get_help_directories", lambda: [tmp_path])

    assert cli._get_help_url(words) == f"help:user/commands/a&amp;b.html{suffix}"
