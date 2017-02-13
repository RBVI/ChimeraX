# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def usage(session, command_name=None, option=None):
    '''Display command usage.

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
    show_hidden = option == 'allOptions'
    if command_name is None:
        info("Use 'usage <command>' for a command synopsis.")
        info("Use 'help <command>' to learn more about a command.")
        cmds = cli.registered_commands(multiword=True)
        if len(cmds) > 0:
            text = cli.commas(cmds, ' and')
            noun = cli.plural_form(cmds, 'command')
            verb = cli.plural_form(cmds, 'is', 'are')
            info("The following %s %s available: %s" % (noun, verb, text))
        return
    elif command_name == 'all':
        info("Syntax for all commands:")
        cmds = cli.registered_commands(multiword=True)
        if not session.ui.is_gui:
            for name in cmds:
                try:
                    info(cli.usage(name, show_subcommands=False, expand_alias=False,
                                   show_hidden=allOptions))
                except:
                    info('%s -- no documentation' % name)
            return
        for name in cmds:
            try:
                info(cli.html_usage(name, show_subcommands=False, expand_alias=False,
                                    show_hidden=allOptions), is_html=True)
            except:
                from html import escape
                info('<b>%s</b> &mdash; no documentation' % escape(name),
                     is_html=True)
        return

    try:
        usage = cli.usage(command_name, show_hidden=show_hidden)
    except ValueError as e:
        from ..errors import UserError
        raise UserError(str(e))
    if session.ui.is_gui:
        info(cli.html_usage(command_name, show_hidden=show_hidden), is_html=True)
    else:
        info(usage)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[
            ('option',
                    cli.Or(cli.EnumOf(['allOptions'], abbreviations=False),
                           cli.EmptyArg)),
            ('command_name', cli.RestOfLine)
        ],
       non_keyword=['command_name', 'option'],
       hidden=['option'],
       synopsis='show command usage')
    cli.register('usage', desc, usage)
