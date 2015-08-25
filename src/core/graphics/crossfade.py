# vi: set expandtab shiftwidth=4 softtabstop=4:
# Allow fading from one scene to another

from .drawing import Drawing


class CrossFade(Drawing):
    '''
    Fade between one rendered scene and the next scene.
    This is a Drawing overlay that is added to the View to cause the
    cross-fade to be rendered.  It is automatically added to the View
    when constructed and removes itself from the View when the fade
    is complete.  '''
    def __init__(self, viewer, frames):

        Drawing.__init__(self, 'cross fade')
        self.viewer = viewer
        self.frames = frames
        self.frame = 0
        self.rgba = None

        self.capture_image()

    def capture_image(self):

        # Capture current image
        v = self.viewer
        self.rgba = v.frame_buffer_rgba()

        # Make textured square surface piece
        from .drawing import rgba_drawing
        rgba_drawing(self.rgba, (-1, -1), (2, 2), self)

        v.add_overlay(self)
        v.add_callback('new frame', self.next_frame)

    def next_frame(self):

        f, n = self.frame + 1, self.frames
        if f >= n:
            v = self.viewer
            v.remove_callback('new frame', self.next_frame)
            v.remove_overlays([self])
            self.remove_all_drawings()
            return

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
        self.capture_image()

    def draw(self, renderer, place, draw_pass, selected_only=False):
        if draw_pass == self.OPAQUE_DRAW_PASS:
            self.changed = self.capture_image()
        elif self.changed:
            Drawing.draw(self, renderer, place, draw_pass, selected_only)

    def capture_image(self):

        self.redraw_needed()

        # Capture current image
        v = self.viewer
        w, h = v.window_size
        rgba = v.frame_buffer_rgba()

        if self.rgba is None:
            self.rgba = rgba
            # Make textured square surface piece
            from .drawing import rgba_drawing
            self.piece = rgba_drawing(rgba, (-1, -1), (2, 2), self)
            v.add_overlay(self)
        elif self.rgba.shape != (h, w, 4):
            # Resize texture and motion blur image
            self.remove_drawing(self.piece)
            from .drawing import rgba_drawing
            self.piece = rgba_drawing(rgba, (-1, -1), (2, 2), self)
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
            self.piece.texture.reload_texture(self.rgba)
        self.redraw_needed()
        return True

    def delete(self):
        Drawing.delete(self)
