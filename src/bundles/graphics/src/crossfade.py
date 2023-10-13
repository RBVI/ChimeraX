# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# Allow fading from one scene to another

from .drawing import Drawing
from chimerax.core.errors import LimitationError


class CrossFade(Drawing):
    '''
    Fade between one rendered scene and the next scene.
    This is a Drawing overlay that is added to the View to cause the
    cross-fade to be rendered.  It is automatically added to the View
    when constructed and removes itself from the View when the fade
    is complete.  '''
    def __init__(self, session, frames):

        Drawing.__init__(self, 'cross fade')
        self.frames = frames
        self.frame = 0
        self.rgba = None

        has_graphics = session.main_view.render is not None
        if not has_graphics:
            raise LimitationError("Unable to do crossfade without rendering images")
        self.capture_image(session)

    def capture_image(self, session):

        # Capture current image
        v = session.main_view
        if hasattr(v, 'movie_image_rgba'):
            self.rgba = v.movie_image_rgba # Recording a movie.  Use its last image.
        else:
            v.render.make_current()
            self.rgba = v.frame_buffer_rgba()

        # Make textured square surface piece
        from .drawing import rgba_drawing
        rgba_drawing(self, self.rgba, (-1, -1), (2, 2))
        self.opaque_texture = False

        v.add_overlay(self)
        session.triggers.add_handler('new frame', lambda *_, v=v: self.next_frame(v))

    def next_frame(self, view):

        f, n = self.frame + 1, self.frames
        if f >= n:
            view.remove_overlays([self])
            self.remove_all_drawings()
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER

        # Increase texture transparency
        self.frame = f
        alpha = int(255 * (n - f) / n)
        self.rgba[:, :, 3] = alpha
        self.texture.reload_texture(self.rgba)
        self.redraw_needed()


class MotionBlur(Drawing):
    '''
    Leave faint images of previous rendered frames as the camera is moved
    through a scene.  This is a Drawing overlay that is added to the View
    to render the motion blur.  It is added to the View by the constructor
    and it can be removed from the View to stop the motion blur effect.
    '''
    def __init__(self, viewer):

        Drawing.__init__(self, 'motion blur')
        self.viewer = viewer
        self.rgba = None
        self.decay_factor = 0.9
        '''
        The Nth previous rendered frame is dimmed by the decay factor to
        the Nth power.  The dimming is achieved by fading to the current
        background color.
        '''
        self.attenuate = 0.5
        "All preceding frames are additionally dimmed by this factor."
        self.changed = True

        has_graphics = viewer.render is not None
        if not has_graphics:
            raise LimitationError("Unable to do motion blur without rendering images")
        self.capture_image()

    def draw(self, renderer, draw_pass):
        if draw_pass == self.OPAQUE_DRAW_PASS:
            self.changed = self.capture_image()
        elif self.changed:
            Drawing.draw(self, renderer, draw_pass)

    def capture_image(self):

        self.redraw_needed()

        # Capture current image
        v = self.viewer
        w, h = v.window_size
        v.render.make_current()
        rgba = v.frame_buffer_rgba()

        if self.rgba is None:
            self.rgba = rgba
            # Make textured square surface piece
            from .drawing import rgba_drawing
            rgba_drawing(self, rgba, (-1, -1), (2, 2))
            v.add_overlay(self)
        elif rgba.shape != self.rgba.shape:
            # Resize texture and motion blur image
            from .drawing import rgba_drawing
            rgba_drawing(self, rgba, (-1, -1), (2, 2))
            self.rgba = rgba
        else:
            from numpy import array
            bgcolor = array([255 * c for c in v.background_color[:3]],
                            rgba.dtype)
            alpha = 255 * self.attenuate
            from ._graphics import blur_blend_images
            c = blur_blend_images(self.decay_factor, rgba, self.rgba,
                                  bgcolor, alpha, self.rgba)
            if c == 0:
                return False    # No change
            self.texture.reload_texture(self.rgba)
        self.opaque_texture = False
        self.redraw_needed()
        return True

    def delete(self):
        Drawing.delete(self)
