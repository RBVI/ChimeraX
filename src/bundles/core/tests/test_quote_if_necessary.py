"""quote_if_necessary names unusual whitespace without importing unicodedata,
an extension module, for ordinary strings (menus quote names at startup)."""
import subprocess
import sys


def test_unusual_whitespace_is_named():
    from chimerax.core.commands.cli import quote_if_necessary
    assert quote_if_necessary('a b') == '"a\\N{EM SPACE}b"'
    assert quote_if_necessary('two words') == '"two words"'
    assert quote_if_necessary('plain') == 'plain'


def test_ordinary_strings_do_not_import_unicodedata():
    code = ("import sys\n"
            "from chimerax.core.commands.cli import quote_if_necessary\n"
            "sys.modules.pop('unicodedata', None)\n"
            "quote_if_necessary('two words'); quote_if_necessary('tab\\there')\n"
            "print('unicodedata' in sys.modules)\n")
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert out.stdout.split() == ['False'], out.stderr
