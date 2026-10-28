from types import SimpleNamespace
import builtins
import pytest


@pytest.mark.parametrize('name,multiple', [('OpenFileNameArg', False), ('OpenFileNamesArg', True),
    ('SaveFileNameArg', False), ('OpenFolderNameArg', False), ('SaveFolderNameArg', False)])
def test_literal_path_does_not_import_qt(name, multiple, tmp_path, monkeypatch):
    from chimerax.core.commands import cli
    path = tmp_path if 'Folder' in name else tmp_path / 'file with spaces.txt'
    if 'Folder' not in name:
        path.write_text('data')
    original = builtins.__import__
    def importing(module, *args, **kwargs):
        if module == 'Qt' or module.startswith('Qt.'):
            raise AssertionError('Literal file arguments must not load Qt')
        return original(module, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    value, text, rest = getattr(cli, name).parse('"' + str(path) + '"',
        SimpleNamespace(ui=SimpleNamespace(is_gui=True)))
    assert value == ([str(path)] if multiple else str(path))
    assert rest == ''


def test_headless_browse_keeps_useful_error():
    from chimerax.core.commands.cli import OpenFileNameArg, AnnotationError
    with pytest.raises(AnnotationError, match='nogui'):
        OpenFileNameArg.parse('browse', SimpleNamespace(ui=SimpleNamespace(is_gui=False)))


@pytest.mark.parametrize('accept,mode', [('AcceptOpen', 'ExistingFile'),
    ('AcceptOpen', 'ExistingFiles'), ('AcceptSave', 'AnyFile'),
    ('AcceptOpen', 'Directory'), ('AcceptSave', 'Directory')])
def test_browse_resolves_modes_on_demand(accept, mode, tmp_path, monkeypatch):
    import sys
    from chimerax.core.commands.cli import _browse_parse
    values = {}
    accept_modes = SimpleNamespace(AcceptOpen=object(), AcceptSave=object())
    file_modes = SimpleNamespace(ExistingFile=object(), ExistingFiles=object(),
                                AnyFile=object(), Directory=object())
    class Dialog:
        AcceptMode = accept_modes
        FileMode = file_modes
        def setAcceptMode(self, value): values['accept'] = value
        def setFileMode(self, value): values['mode'] = value
        def setNameFilter(self, value): values['filter'] = value
        def exec(self): return True
        def selectedFiles(self): return [str(tmp_path)]
    monkeypatch.setitem(sys.modules, 'Qt.QtWidgets', SimpleNamespace(QFileDialog=Dialog))
    value, text, rest = _browse_parse('browse',
        SimpleNamespace(ui=SimpleNamespace(is_gui=True)), 'file', '*', accept, mode, True)
    assert value == str(tmp_path)
    assert values == {'accept': getattr(accept_modes, accept),
                      'mode': getattr(file_modes, mode), 'filter': '*'}
