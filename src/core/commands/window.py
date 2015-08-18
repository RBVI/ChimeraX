# vi: set expandtab shiftwidth=4 softtabstop=4:

def window(session):
    '''Move camera so the displayed models fill the graphics window.'''
    session.main_view.view_all()

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='reset view so everything is visible in window')
    cli.register('window', desc, window)
