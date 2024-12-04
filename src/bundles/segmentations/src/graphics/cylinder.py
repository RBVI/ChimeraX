# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from numpy import array, float32, uint8, int32

from chimerax.shape.shape import cylinder_geometry, _cylinder_divisions
from chimerax.surface import calculate_vertex_normals
from chimerax.core.models import Model
from chimerax.graphics import Drawing
from chimerax.geometry import Place


class SegmentationDisk(Model):
    SESSION_ENDURING = True
    SESSION_SAVE = False

    def __init__(self, session, axis, radius=10, height=10, divisions=50):
        super().__init__(" ".join([str(axis), "segmentation cursor"]), session)
        self.display_style = Drawing.Solid
        self.use_lighting = True
        self.axis = axis
        self._radius = radius
        self._height = (
            height  # set to the slice thickness of whatever DICOM is being observed
        )
        self._divisions = divisions
        self.color = [255, 85, 0, 255]
        self._slice = 1
        self.position = Place(axes=axis.transform, origin=[0, 0, 0])
        self.update_geometry()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius
        self.update_geometry()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height
        self.update_geometry()

    @property
    def divisions(self):
        return self._divisions

    @divisions.setter
    def divisions(self, divisions):
        self._divisions = divisions
        self.update_geometry()

    #    @property
    #    def slice(self):
    #        return self._slice
    #
    #    @slice.setter
    #    def slice(self, slice):
    #        self._slice = slice

    @property
    def origin(self):
        return self.position.origin()

    @origin.setter
    def origin(self, origin):
        self.position = Place(axes=self.axis.transform, origin=origin)

    def update_geometry(self):
        nz, nc = _cylinder_divisions(self.radius, self.height, self.divisions)
        varray, tarray = cylinder_geometry(self.radius, self.height, nz, nc, caps=True)
        narray = calculate_vertex_normals(varray, tarray)
        self.set_geometry(varray, narray, tarray)
