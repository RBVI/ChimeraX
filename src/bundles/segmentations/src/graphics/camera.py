# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from chimerax.geometry import Place
from chimerax.graphics import Camera


class OrthoCamera(Camera):
    """A limited camera for the Side View without field_of_view"""

    def __init__(self):
        Camera.__init__(self)
        self.position = Place()
        self.near = 0.1
        self.far = 100000000
        self.field_width = 1

    def get_position(self, view_num=None):
        return self.position

    def number_of_views(self):
        return 1

    def combine_rendered_camera_views(self, render):
        return

    def projection_matrix(self, near_far_clip, view_num, window_size):
        near, far = near_far_clip
        ww, wh = window_size
        aspect = wh / ww
        w = self.field_width
        h = w * aspect
        left, right, bot, top = -0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h
        from chimerax.graphics.camera import ortho

        pm = ortho(left, right, bot, top, near, far)
        return pm

    def view_width(self, center):
        return self.field_width
