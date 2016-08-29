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

def stop(session):
    '''Stop a motion initiated with turn, roll or move with the frames argument.'''
    from .motion import CallForNFrames
    has_motion = False
    if hasattr(session, CallForNFrames.Attribute):
        for mip in tuple(getattr(session, CallForNFrames.Attribute)):
            has_motion = True
            mip.done()
    if not has_motion:
        session.logger.status('No motion in progress.  Use "exit" or "quit" to stop program.')


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='stop all motion')
    cli.register('stop', desc, stop)
