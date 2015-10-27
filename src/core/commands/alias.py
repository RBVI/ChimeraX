from . import cli


def alias(session, name='', text=''):
    """Create command alias

    :param name: optional name of the alias
    :param text: optional text of the alias

    If the alias name is not given, then a text list of all the aliases is
    shown.  If alias text is not given, the text of the named alias
    is shown.  If both arguments are given, then a new alias is made.
    """
    logger = session.logger
    if not name:
        # list aliases
        aliases = cli.list_aliases()
        names = cli.commas(aliases, ' and')
        noun = cli.plural_form(aliases, 'alias')
        if names:
            logger.info('%d %s: %s' % (len(aliases), noun, names))
        else:
            logger.status('No aliases.')
        return
    if not text:
        text = cli.expand_alias(name)
        if text is None:
            logger.status('No alias named %s found.' % cli.dq_repr(name))
        else:
            logger.info('Aliased %s to: %s' % (cli.dq_repr(name), text))
        return
    cli.create_alias(name, text, user=True, logger=session.logger)


def unalias(session, name=None):
    """Remove command alias

    :param name: optional name of the alias
        If not given, then remove all aliases.
    """
    cli.remove_alias(name)


def register_command(session):
    desc = cli.CmdDesc(optional=[('name', cli.StringArg),
                                 ('text', cli.WholeRestOfLine)],
                       non_keyword=['name', 'text'],
                       synopsis='list or define a command alias')
    cli.register('alias', desc, alias)
    desc = cli.CmdDesc(optional=[('name', cli.StringArg)],
                       non_keyword=['name'],
                       synopsis='remove a command alias')
    cli.register('~alias', desc, unalias)
