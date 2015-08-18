# vi: set expandtab shiftwidth=4 softtabstop=4:

def help(session, command_name=None):
    '''Display help.

    Parameters
    ----------
    command_name : string
        Show documentation for the specified command.  If no command is specified
        then the names of all commands are shown.  If the command name "all" is
        given then a synopsis for each command is shown.  The command name can
        be abbreviated.
    '''
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
                 % (', '.join(cmds[:-1]), cmds[-1]))
        return
    elif command_name == 'all':
        info("Syntax for all commands.")
        cmds = cli.registered_commands()
        cmds.sort()
        for name in cmds:
            try:
                info(cli.html_usage(name), is_html=True)
            except:
                info('<b>%s</b> no documentation' % name, is_html=True)
        return

    try:
        usage = cli.usage(command_name)
    except ValueError as e:
        status(str(e))
        return
    if session.ui.is_gui:
        info(cli.html_usage(command_name), is_html=True)
    else:
        info(usage)

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(optional=[('command_name', cli.RestOfLine)],
                         synopsis='show command usage')
    cli.register('help', desc, help)
