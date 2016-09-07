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

def window_size(session, width=None, height=None):
    '''Report or set graphics window size in pixels.'''

    v = session.main_view
    w, h = v.window_size
    if width is None and height is None:
        msg = 'window size %d %d' % (w, h)
        log = session.logger
        log.status(msg)
        log.info(msg)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
        if not session.ui.is_gui:
            v.window_size = width, height
        elif width != w or height != h:
            session.ui.main_window.adjust_size(width-w, height-h)

def register_command(session):
    from . import CmdDesc, register, PositiveIntArg
    desc = CmdDesc(optional=[('width', PositiveIntArg),
                             ('height', PositiveIntArg)],
                   synopsis='report or set window size')
    register('windowsize', desc, window_size)
