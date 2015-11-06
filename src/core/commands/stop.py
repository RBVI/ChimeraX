# vim: set expandtab shiftwidth=4 softtabstop=4:


def stop(session):
    '''Stop a motion initiated with turn, roll or move with the frames argument.'''
    from .motion import CallForNFrames
    has_motion = False
    if hasattr(session, CallForNFrames.Attribute):
        for mip in tuple(getattr(session, CallForNFrames.Attribute)):
            has_motion = True
            mip.done()
    if not has_motion:
        session.logger.status('No motion in progress.  Use "exit" to stop program.')


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='stop all motion')
    cli.register('stop', desc, stop)
