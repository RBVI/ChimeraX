# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
commands -- Default set of commands
===================================

This module implements a default set of cli commands.
After importing this module, :py:func:`register`
must be called to get the commands recognized by the command line interface
(:py:mod:`chimera2.cli`).
"""

from . import cli


def pwd(session):
    import os
    session.logger.info('current working directory: %s' % os.getcwd())
_pwd_desc = cli.CmdDesc()


def exit(session):
    session.ui.quit()
_exit_desc = cli.CmdDesc()


def stop(session, ignore=None):
    raise cli.UserError('use "exit" or "quit" instead of "stop"')
_stop_desc = cli.CmdDesc(optional=[('ignore', cli.RestOfLine)])


def echo(session, text=''):
    session.logger.info(text)
_echo_desc = cli.CmdDesc(optional=[('text', cli.RestOfLine)])


def open(session, filename, id=None, as_=None):
    try:
        return session.models.open(filename, id=id, name=name)
    except OSError as e:
        raise cli.UserError(e)
_open_desc = cli.CmdDesc(required=[('filename', cli.StringArg)],
                         keyword=[('id', cli.ModelIdArg),
                                  ('as', cli.StringArg)])


def export(session, filename, **kw):
    try:
        from . import io
        return io.export(session, filename, **kw)
    except OSError as e:
        raise cli.UserError(e)
_export_desc = cli.CmdDesc(required=[('filename', cli.StringArg)])


def close(session, model_ids):
    try:
        for model_id in model_ids:
            session.models.close(model_id)
    except ValueError as e:
        raise cli.UserError(e)
_close_desc = cli.CmdDesc(required=[('model_ids', cli.ListOf(cli.ModelIdArg))])


def list(session):
    models = session.models.list()
    if len(models) == 0:
        session.logger.status("No open models.")
        return

    def id_str(id):
        if isinstance(id, int):
            return str(id)
        return '.'.join(str(x) for x in id)
    ids = [m.id for m in models]
    ids.sort()
    info = "Open models: "
    if len(models) > 1:
        info += ", ".join(id_str(id) for id in ids[:-1]) + " and"
    info += " %s" % id_str(ids[-1])
    session.logger.info(info)
_list_desc = cli.CmdDesc()


def help(session, command_name=None):
    from . import cli
    status = session.logger.status
    info = session.logger.info
    if command_name is None:
        info("Use 'help <command>' to learn more about a command.")
        cmds = cli.registered_commands()
        cmds.sort()
        if len(cmds) == 0:
            pass
        elif len(cmds) == 1:
            info("The following command is available: %s" % cmds[0])
        else:
            info("The following commands are available: %s, and %s"
                 % ( ', '.join(cmds[:-1]), cmds[-1]))
        return
    status(cli.usage(command_name))
    info(cli.html_usage(command_name), is_html=True)
_help_desc = cli.CmdDesc(optional=[('command_name', cli.StringArg)])


def window(session):
    session.main_view.view_all()
_window_desc = cli.CmdDesc()


def register(session):
    """Register common cli commands"""
    cli.register('exit', _exit_desc, exit)
    cli.alias(session, "quit", "exit $*")
    cli.register('open', _open_desc, open)
    cli.register('close', _close_desc, close)
    cli.register('export', _export_desc, export)
    cli.register('list', _list_desc, list)
    cli.register('stop', _stop_desc, stop)
    cli.register('echo', _echo_desc, echo)
    cli.register('pwd', _pwd_desc, pwd)
    cli.register('window', _window_desc, window)
    cli.register('help', _help_desc, help)
    from . import molsurf
    molsurf.register_surface_command()
    from . import structure
    structure.register_molecule_commands()
    from . import lightcmd
    lightcmd.register_lighting_command()
    from . import map
    map.register_volume_command()
    from .map import series
    series.register_vseries_command()
    from . import color
    color.register_commands()

    # def lighting_cmds():
    #     import .lighting.cmd as cmd
    #     cmd.register()
    # cli.delay_registration('lighting', lighting_cmds)
