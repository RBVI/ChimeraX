# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def crossfade(session, frames=30):
    '''Fade from the current view to the next drawn view. Used in movie recording.

    Parameters
    ----------
    frames : integer
        Linear interpolate between the current and next image over this number of frames.
    '''
    from chimerax.core.graphics import CrossFade
    CrossFade(session, frames)


def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, PositiveIntArg
    desc = CmdDesc(
        optional=[('frames', PositiveIntArg)],
        synopsis='Fade between one rendered scene and the next scene.')
    register('crossfade', desc, crossfade, logger=logger)
