# vi: set expandtab shiftwidth=4 softtabstop=4:

def stop(session):
    '''Stop a motion initiated with turn, roll or move with the frames argument.'''
    from .motion import CallForNFrames
    if not hasattr(session, CallForNFrames.Attribute):
        return
    for mip in tuple(getattr(session, CallForNFrames.Attribute)):
        mip.done()

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='stop all motion')
    cli.register('stop', desc, stop)
