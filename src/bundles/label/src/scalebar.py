# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
#
def scalebar(session, *, length = None, height = None, color = None, xpos = None, ypos = None):
    '''
    Create a scalebar label.

    Parameters
    ----------
    length : float
      Length in physical units (Angstroms).  Default 100.
    height : float
      Thickness of scalebar in pixels.  Default 10.
    color : Color
      Color of the scalebar.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    xpos : float
      Placement of left edge of scalebar. Range 0 - 1 covers full width of graphics window.
      Default 0.1
    ypos : float
      Placement of bottom edge of scalebar. Range 0 - 1 covers full height of graphics window.
      Default 0.1
    '''
    name = 'scalebar'
    from . import label2d
    lm = label2d.session_labels(session)
    label = lm.named_label(name) if lm else None
    if label is None:
        x = 0.1 if xpos is None else xpos
        y = 0.1 if ypos is None else ypos
        label = label2d.label_create(session, name, color=color, xpos=x, ypos=y)
        label.scalebar_width = 100
        label.scalebar_height = 10

    if length is not None:
        label.scalebar_width = length
    if height is not None:
        label.scalebar_height = height
    if color is not None or xpos is not None or ypos is not None:
        label2d._update_label(session, label, color=color, xpos=xpos, ypos=ypos)

    return label

# -----------------------------------------------------------------------------
#
def scalebar_off(session):
    '''
    Delete the scalebar label.
    '''
    from .label2d import session_labels
    lm = session_labels(session)
    if lm:
        label = lm.named_label('scalebar')
        if label:
            label.delete()

scalebar_delete = scalebar_off

# -----------------------------------------------------------------------------
#
def register_scalebar_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, ColorArg, Or, EnumOf
    desc = CmdDesc(optional = [('length', FloatArg)],
                   keyword = [('height', FloatArg),
                              ('color', Or(EnumOf(['default', 'auto']), ColorArg)),
                              ('xpos', FloatArg),
                              ('ypos', FloatArg)],
                   synopsis = 'Create or modify a scalebar')
    register('scalebar', desc, scalebar, logger=logger)

    desc = CmdDesc(synopsis = 'Delete scalebar')
    register('scalebar off', desc, scalebar_off, logger=logger)
    desc = CmdDesc(synopsis = 'Delete scalebar')
    register('scalebar delete', desc, scalebar_off, logger=logger)
