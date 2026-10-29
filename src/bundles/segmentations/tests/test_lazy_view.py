from types import SimpleNamespace


def test_view_class_is_resolved_on_demand(monkeypatch):
    import sys
    from chimerax.segmentations import view
    view_class = object()
    monkeypatch.delattr(view, 'FourPanelView', raising=False)
    monkeypatch.setitem(sys.modules, view.__name__ + '.ui',
                        SimpleNamespace(FourPanelView=view_class))
    assert 'FourPanelView' in dir(view)
    assert view.FourPanelView is view_class
    assert view.FourPanelView is view_class
    assert view.views == ['fourup', 'sidebyside', 'overunder']


def test_layout_command_resolves_view_when_creating_it(monkeypatch):
    from chimerax.segmentations import view
    from chimerax.segmentations.view import cmd
    calls = []
    panel = SimpleNamespace(register_segmentation_tool=lambda tool: calls.append(tool))
    monkeypatch.setattr(view, 'FourPanelView',
                        lambda session, layout: calls.append(layout) or panel)
    monkeypatch.setattr(cmd, 'find_segmentation_tool', lambda session: None)
    monkeypatch.setattr(cmd, 'get_settings', lambda session: SimpleNamespace())
    monkeypatch.setattr(cmd.chimerax.segmentations.triggers, 'activate_trigger',
                        lambda *args: None)
    window = SimpleNamespace(view_layout='default')
    session = SimpleNamespace(ui=SimpleNamespace(main_window=window))
    cmd.view_layout(session, 'fourup')
    assert calls == ['fourup']
    assert window.main_view is panel
    assert window.view_layout == 'orthoplanes'
