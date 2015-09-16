# vi: set expandtab shiftwidth=4 softtabstop=4:


def pwd(session):
    '''Report the current directory to the log.'''
    import os
    session.logger.info('current working directory: %s' % os.getcwd())


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='print current working directory')
    cli.register('pwd', desc, pwd)
