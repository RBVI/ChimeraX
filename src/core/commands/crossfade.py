# vim: set expandtab shiftwidth=4 softtabstop=4:


def crossfade(session, frames=30):
    '''Fade from the current view to the next drawn view. Used in movie recording.

    Parameters
    ----------
    frames : integer
        Linear interpolate between the current and next image over this number of frames.
    '''
    from ..graphics import CrossFade
    CrossFade(session, frames)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('frames', cli.PositiveIntArg)],
        synopsis='Fade between one rendered scene and the next scene.')
    cli.register('crossfade', desc, crossfade)
