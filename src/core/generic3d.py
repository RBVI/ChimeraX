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

from .models import Surface

from . import toolshed
CATEGORY = toolshed.GENERIC3D

# If any of the *STATE_VERSIONs change, then increase the (maximum) core session
# number in setup.py.in
GENERIC3D_STATE_VERSION = 1


class Generic3DModel(Surface):
    """Commom base class for generic 3D data"""

    def take_snapshot(self, session, flags):
        from chimerax.graphics.gsession import DrawingState
        data = {
            'model state': Surface.take_snapshot(self, session, flags),
            'drawing state': DrawingState().take_snapshot(self, session, flags),
            'version': GENERIC3D_STATE_VERSION,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        m = cls('name', session)
        m.set_state_from_snapshot(session, data['model state'])
        from chimerax.graphics.gsession import DrawingState
        DrawingState().set_state_from_snapshot(m, session, data['drawing state'])
        return m

    def geometry_bounds(self):
        '''
        Unlike Drawing.geometry_bounds, return union of bounds of non-Model children.
        '''
        from chimerax.geometry import union_bounds, copies_bounding_box
        from .models import Model
        bm = [copies_bounding_box(d.geometry_bounds(), d.get_scene_positions())
              for d in self.child_drawings() if not isinstance(d, Model)]
        bm.append(super().geometry_bounds())  # get bounds of any geometry in this drawing
        b = union_bounds(bm)
        return b
