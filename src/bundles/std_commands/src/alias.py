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

from chimerax.core.commands import cli


def alias(session, name, text=''):
    """Create command alias

    :param name: name of the alias
    :param text: optional text of the alias

    If the alias name is not given, then a text list of all the aliases is
    shown.  If alias text is not given, the text of the named alias
    is shown.  If both arguments are given, then a new alias is made.
    """
    logger = session.logger
    if not text:
        text = cli.expand_alias(name)
        if text is None:
            logger.status('No alias named %s found.' % cli.dq_repr(name))
        else:
            logger.info('Aliased %s to: %s' % (cli.dq_repr(name), text))
        return
    cli.create_alias(name, text, user=True, logger=logger)


def list_aliases(session, internal=False):
    # list aliases
    logger = session.logger
    aliases = cli.list_aliases(all=internal, logger=logger)
    aliases.sort(key=lambda x: x[x[0] == '~':])
    names = cli.commas(aliases, 'and')
    noun = cli.plural_form(aliases, 'alias')
    if names:
        logger.info('%d %s: %s' % (len(aliases), noun, names))
    else:
        logger.status('No %saliases.' % ('custom ' if not internal else ''))
    return


def unalias(session, name):
    """Remove command alias

    :param name: optional name of the alias
        If not given, then remove all aliases.
    """
    logger = session.logger
    if name == 'all':
        cli.remove_alias(user=True, logger=logger)
    else:
        cli.remove_alias(name, user=True, logger=logger)


def register_command(logger):
    desc = cli.CmdDesc(
        required=[('name', cli.StringArg)],
        optional=[('text', cli.WholeRestOfLine)],
        non_keyword=['text'],
        synopsis='define or show a command alias')
    cli.register('alias', desc, alias, logger=logger)

    desc = cli.CmdDesc(
        keyword=[('internal', cli.NoArg)],
        synopsis='list aliases')
    cli.register('alias list', desc, list_aliases, logger=logger)

    desc = cli.CmdDesc(
        required=[('name', cli.Or(cli.EnumOf(['all']), cli.StringArg))],
        non_keyword=['name'],
        synopsis='remove a command alias')
    cli.register('alias delete', desc, unalias, logger=logger)

    cli.create_alias('~alias', 'alias delete $*', logger=logger)
