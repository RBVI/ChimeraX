from types import SimpleNamespace

import pytest

from chimerax.core.commands import cli
from chimerax.core.session import register_misc_commands


@pytest.fixture
def management_commands(monkeypatch):
    from chimerax.core.commands import devel, pip, toolshed

    monkeypatch.setattr(cli, '_command_info', cli.RegisteredCommandInfo())
    # The application fixture may already have registered these descriptors.
    # Give this isolated command registry its own unbound copies.
    for module in (devel, pip, toolshed):
        for name, value in list(vars(module).items()):
            if isinstance(value, cli.CmdDesc):
                monkeypatch.setattr(module, name, value.copy())
    calls = []
    for module in (devel, pip, toolshed):
        register = module.register_command
        def tracked(logger, module=module, register=register):
            calls.append(module.__name__.rsplit('.', 1)[-1])
            register(logger)
        monkeypatch.setattr(module, 'register_command', tracked)
    register_misc_commands(SimpleNamespace(logger=None))
    return calls


@pytest.mark.parametrize('first', ['pip', 'devel pip'])
def test_both_pip_spellings_register_once(management_commands, first):
    calls = management_commands
    assert calls == []
    cli.command_function(first)
    pip = cli.command_function('pip')
    alias = cli.command_function('devel pip')
    assert pip.__name__ == 'pip'
    assert isinstance(alias, cli.Alias)
    assert alias.expand(optional='show numpy') == 'pip show numpy'
    assert calls.count('pip') == 1
    assert calls.count('devel') == 1
    assert 'toolshed' not in calls


def test_usage_completion_and_neighboring_toolshed_command(management_commands):
    # A bundle's toolshed subcommand must survive registration of the core group.
    def updates(session):
        pass
    cli.register('toolshed updates', cli.CmdDesc(synopsis='updates'), updates)
    assert cli.command_function('toolshed updates') is updates
    assert cli.command_function('toolshed url').__name__ == 'toolshed_url'
    assert cli.command_function('devel al').__name__ == 'devel_alias'
    syntax = cli.usage(None, 'pip')
    assert 'install' in syntax and 'verbose' in syntax
    names = cli.registered_commands(multiword=True)
    for name in ('toolshed updates', 'toolshed list', 'devel alias', 'devel build',
                 'devel pip', 'pip', 'debug sdump', 'debug exception'):
        assert name in names
    assert management_commands.count('toolshed') == 1
