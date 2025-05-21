# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def move_cofr(session, models):
    '''
    Translate all the models as a rigid group so their bounding box center
    is at the current center of rotation.
    '''
    from chimerax.geometry import union_bounds, translation
    b = union_bounds(m.bounds() for m in models)
    if b is None:
        from chimerax.core.errors import UserError
        raise UserError('Could not compute center of models since none are displayed')
    mcenter = b.center()
    cofr = session.main_view.center_of_rotation
    translate = translation(cofr - mcenter)
    
    from .view import UndoView
    undo = UndoView("move cofr", session, models)
    with session.undo.block():
        for m in models:
            m.scene_position = translate * m.scene_position
    undo.finish(session, models)
    session.undo.register(undo)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, TopModelsArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('models', TopModelsArg)],
        synopsis='move models to center of rotation'
    )
    register('move cofr', desc, move_cofr, logger=logger)
