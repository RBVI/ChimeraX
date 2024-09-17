# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2023 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from chimerax.core.models import Surface


class SegmentationSphere(Surface):
    SESSION_SAVE = False

    def __init__(self, name, session, center=[0, 0, 0], radius=10):
        self._num_triangles = 1000
        Surface.__init__(self, name, session)
        from chimerax.surface import sphere_geometry2

        va, na, ta = sphere_geometry2(self._num_triangles)
        self._unit_vertices = va
        self.set_geometry(radius * va, na, ta)
        self.color = [255, 85, 0, 255]
        from chimerax.geometry import translation

        self.position = translation(center)
        self._radius = radius
        session.models.add([self])

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        self.set_geometry(r * self._unit_vertices, self.normals, self.triangles)
