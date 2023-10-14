# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
    from chimerax.core.commands import cli
    info = session.logger.info
    show_hidden = option == 'allOptions'
    if command_name is None:
        info("Use 'usage <command>' for a command synopsis.")
        info("Use 'help <command>' to learn more about a command.")
        cmds = cli.registered_commands(multiword=True)
        if len(cmds) > 0:
            text = cli.commas(cmds, 'and')
            noun = cli.plural_form(cmds, 'command')
            verb = cli.plural_form(cmds, 'is', 'are')
            info("The following %s %s available: %s" % (noun, verb, text))
        return
    elif command_name == 'all':
        info("Syntax for all commands:")
        usage = []
        cmds = cli.registered_commands(multiword=True)
        if not session.ui.is_gui:
            for name in cmds:
                try:
                    usage.append(cli.usage(
                        session, name, show_subcommands=0, expand_alias=False,
                        show_hidden=show_hidden))
                except Exception:
                    usage.append('%s -- no documentation' % name)
            info('\n'.join(usage))
            return
        for name in cmds:
            try:
                usage.append(cli.html_usage(
                    session, name, show_subcommands=0, expand_alias=False,
                    show_hidden=show_hidden))
            except Exception:
                #import traceback
                #traceback.print_exc()
                from html import escape
                usage.append('<b>%s</b> &mdash; no documentation' % escape(name)
)
        info('<p>'.join(usage), is_html=True)
        return

    try:
        if session.ui.is_gui:
            usage = cli.html_usage(session, command_name, show_hidden=show_hidden)
        else:
            usage = cli.usage(session, command_name, show_hidden=show_hidden)
    except ValueError as e:
        session.logger.warning(str(e))
    else:
        info(usage, is_html=session.ui.is_gui)


def register_command(logger):
    from chimerax.core.commands import CmdDesc, Or, EnumOf, EmptyArg, RestOfLine, register
    desc = CmdDesc(
        optional=[
            ('option', Or(EnumOf(['allOptions'], abbreviations=False), EmptyArg)),
            ('command_name', RestOfLine)
        ],
        non_keyword=['command_name', 'option'],
        hidden=['option'],
        synopsis='show command usage')
    register('usage', desc, usage, logger=logger)
