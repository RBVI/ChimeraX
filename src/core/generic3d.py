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

from .models import Model

from . import toolshed
CATEGORY = toolshed.GENERIC3D


class Generic3DModel(Model):
    """Commom base class for generic 3D data"""

    def take_snapshot(self, session, flags):
        from .state import CORE_STATE_VERSION
        from .graphics.gsession import DrawingState
        data = {
            'model state': Model.take_snapshot(self, session, flags),
            'drawing state': DrawingState().take_snapshot(self, session, flags),
            'version': CORE_STATE_VERSION,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        m = cls('name', session)
        m.set_state_from_snapshot(session, data['model state'])
        from .graphics.gsession import DrawingState
        DrawingState().set_state_from_snapshot(m, session, data['drawing state'])
        return m

    def reset_state(self, session):
        pass
