from types import SimpleNamespace


def test_updates_registration_is_deferred(monkeypatch):
    from chimerax.core.commands import cli
    from chimerax.core import toolshed
    from chimerax import toolshed_utils
    from chimerax.toolshed_utils import cmd

    deferred = {}
    registered = []
    handlers = {}
    logger = object()
    session = SimpleNamespace(logger=logger, toolshed=SimpleNamespace(
        triggers=SimpleNamespace(add_handler=lambda name, cb: handlers.update({name: cb}))))
    monkeypatch.setattr(cli, 'delay_registration',
                        lambda name, cb, **kw: deferred.update({name: cb}))
    monkeypatch.setattr(cmd, 'register_command', registered.append)

    toolshed_utils.bundle_api.initialize(session, None)
    assert registered == []
    assert list(deferred) == ['toolshed updates']
    assert toolshed.TOOLSHED_OUT_OF_DATE_BUNDLES in handlers
    deferred['toolshed updates']()
    assert registered == [logger]
