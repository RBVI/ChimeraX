"""History data remains usable before constructing its Qt window."""
from types import SimpleNamespace

import pytest

from chimerax.cmd_line.tool import _HistoryDialog


class Text:
    def __init__(self):
        self.value = ''
        self.items = []
        self.blocked = False

    def blockSignals(self, value):
        old, self.blocked = self.blocked, value
        return old

    def clear(self):
        self.items.clear()
        self.value = ''

    def addItems(self, items):
        self.items.extend(items)
        if items:
            self.value = items[0]

    def lineEdit(self):
        return self

    def setText(self, value):
        self.value = value

    def setFocus(self):
        pass

    def selectAll(self):
        pass


@pytest.fixture
def history(monkeypatch):
    import chimerax.core.history as storage
    text = Text()
    settings = SimpleNamespace(num_remembered=4)
    class MemoryHistory(list):
        def enqueue(self, item):
            self.append(item)
            while len(self) > settings.num_remembered:
                self.pop(0)
    values = MemoryHistory([('first', True), ('generated', False), ('last', True)])
    monkeypatch.setattr(storage, 'FIFOHistory', lambda *args: values)
    def unexpected_window(*args, **kw):
        pytest.fail('History window created before first use')
    controller = SimpleNamespace(
        settings=settings, session=None, text=text, cmd_replace=text.setText,
        show_history_label='Command History', compact_label='Compact History',
        tool_window=SimpleNamespace(create_child_window=unexpected_window))
    return _HistoryDialog(controller, True)


def test_populate_keeps_recent_commands_without_a_window(history):
    history.populate()
    assert history._window is None
    assert history._listbox is None
    assert history.controller.text.items == [
        'last', 'first', 'Command History', 'Compact History']
    assert history.controller.text.value == 'last'
    assert history._history_row == 1


def test_background_recording_preserves_partial_input(history):
    history.populate()
    history.controller.cmd_replace('unfinished')
    history.add('background')
    history.add('more background')  # evicts the oldest typed command
    assert history.history() == ['last']
    assert history._history_row == 0
    assert history.controller.text.value == 'unfinished'
    assert history._window is None


def test_typed_recording_and_filter_changes_stay_lazy(history):
    history.populate()
    history.add('new', typed=True)
    assert history.controller.text.items[:3] == ['new', 'last', 'first']
    assert history._history_row == 2
    history.set_typed_only(False)
    assert history.history() == ['first', 'generated', 'last', 'new']
    assert history._history_row == 3
    assert history._window is None


def test_eviction_of_all_selected_history_leaves_no_selection(history):
    history.populate()
    for i in range(4):
        history.add('background %s' % i)
    assert history.history() == []
    assert history._history_row == -1
    assert history._window is None


def test_window_and_list_access_share_single_creation(history, monkeypatch):
    window, listbox = object(), object()
    calls = []
    def create():
        calls.append(True)
        history._window, history._listbox = window, listbox
    monkeypatch.setattr(history, '_create_window', create)
    assert history.listbox is listbox
    assert history.window is window
    assert calls == [True]


def test_eviction_preserves_absent_selection(history):
    history._history[:] = []
    history.typed_only = False
    history.populate()
    for i in range(6):
        history.add('background %s' % i)
    assert history._history_row == -1
    assert history._window is None
