# vim: set expandtab shiftwidth=4 softtabstop=4:


def exit(session):
    '''Quit the program.'''
    session.ui.quit()


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='exit application')
    cli.register('exit', desc, exit)
    cli.alias("quit", "exit $*")
