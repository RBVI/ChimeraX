# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def scalebar(session, *, length = None, thickness = None, color = None, xpos = None, ypos = None):
    '''
    Create a scalebar label.

    Parameters
    ----------
    length : float
      Length in physical units (Angstroms).  Default 100.
    thickness : float
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
    label = _scalebar_label(session)
    if label is None:
        x = 0.1 if xpos is None else xpos
        y = 0.1 if ypos is None else ypos
        from .label2d import label_create
        label = label_create(session, 'scalebar', color=color, xpos=x, ypos=y)
        label.scalebar_width = 100
        label.scalebar_height = 10

    if length is not None:
        label.scalebar_width = length
    if thickness is not None:
        label.scalebar_height = thickness
    if (length is not None or thickness is not None or
        color is not None or xpos is not None or ypos is not None):
        from .label2d import _update_label
        _update_label(session, label, color=color, xpos=xpos, ypos=ypos)

    return label

# -----------------------------------------------------------------------------
#
def _scalebar_label(session):
    name = 'scalebar'
    from . import label2d
    lm = label2d.session_labels(session)
    label = lm.named_label(name) if lm else None
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
                   keyword = [('thickness', FloatArg),
                              ('color', Or(EnumOf(['default', 'auto']), ColorArg)),
                              ('xpos', FloatArg),
                              ('ypos', FloatArg)],
                   synopsis = 'Create or modify a scalebar')
    register('scalebar', desc, scalebar, logger=logger)

    desc = CmdDesc(synopsis = 'Delete scalebar')
    register('scalebar off', desc, scalebar_off, logger=logger)
    desc = CmdDesc(synopsis = 'Delete scalebar')
    register('scalebar delete', desc, scalebar_off, logger=logger)
