# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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


def alias_usage(session, name, **kw):
    try:
        cli.command_set_alias_usage(name, **kw)
    except ValueError as e:
        session.logger.warning(str(e))


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


class NameDescriptionArg(cli.StringArg):
    name = "an argument name[:description]"

    @staticmethod
    def parse(text, session):
        token, text, rest = cli.StringArg.parse(text, session)
        if not text:
            raise cli.AnnotationError("Alias argument description can't be an empty string")
        if ':' not in token:
            value = (token, None)
        else:
            name, description = token.split(':', maxsplit=1)
            name = name.strip()
            description = description.strip()
            if not name:
                name = None
            if not description:
                description = None
            value = (name, description)
        return (value, text, rest)

    @staticmethod
    def unparse(value, session=None):
        name, description = value
        if name is None:
            name = ''
        if description is None:
            return name
        return f'{name}:{description}'

    @classmethod
    def html_name(cls, name=None):
        return "an argument[<b>:</b>description]"


def register_command(logger):
    desc = cli.CmdDesc(
        required=[('name', cli.StringArg)],
        optional=[('text', cli.WholeRestOfLine)],
        non_keyword=['text'],
        synopsis='define or show a command alias')
    cli.register('alias', desc, alias, logger=logger)

    desc = cli.CmdDesc(
        required=[('name', cli.StringArg)],
        keyword=[
            ('synopsis', cli.StringArg),
            ('url', cli.StringArg),
            ('$1', NameDescriptionArg),
            ('$2', NameDescriptionArg),
            ('$3', NameDescriptionArg),
            ('$4', NameDescriptionArg),
            ('$5', NameDescriptionArg),
            ('$6', NameDescriptionArg),
            ('$7', NameDescriptionArg),
            ('$8', NameDescriptionArg),
            ('$9', NameDescriptionArg),
            ('$*', NameDescriptionArg),
        ],
        synopsis="set alias' usage")
    cli.register('alias usage', desc, alias_usage, logger=logger)
    cli.create_alias('alias synopsis', 'alias usage $1 synopsis "$*"', logger=logger)

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
