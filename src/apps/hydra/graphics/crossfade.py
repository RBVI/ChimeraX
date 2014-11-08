# Allow fading from one scene to another

from .drawing import Drawing
class Cross_Fade(Drawing):
    
    def __init__(self, viewer, frames):

        Drawing.__init__(self, 'cross fade')
        self.viewer = viewer
        self.frames = frames
        self.frame = 0
        self.rgba = None

#        viewer.add_rendered_frame_callback(self.capture_image)
        self.capture_image()

    def capture_image(self):
        
#        viewer.remove_rendered_frame_callback(self.capture_image)

        # Capture current image
        v = self.viewer
        self.rgba = v.frame_buffer_rgba()

        # TODO: Use a childless drawing.
        # Make textured square surface piece
        from .drawing import rgba_drawing
        self.piece = rgba_drawing(self.rgba, (-1,-1), (2,2), self)

        v.add_overlay(self)
        v.add_new_frame_callback(self.next_frame)

    def next_frame(self):

        f,n = self.frame+1, self.frames
        if f >= n:
            v = self.viewer
            v.remove_new_frame_callback(self.next_frame)
            v.remove_overlays([self])
            self.remove_all_drawings()
            return

        # Increase texture transparency
        self.frame = f
        alpha = int(255 * (n-f) / n)
        self.rgba[:,:,3] = alpha
        self.piece.texture.reload_texture(self.rgba)
        self.redraw_needed()

class Motion_Blur(Drawing):
    
    def __init__(self, viewer):

        Drawing.__init__(self, 'motion blur')
        self.viewer = viewer
        self.rgba = None
        self.decay_factor = 0.9
        self.attenuate = 0.5
        self.changed = True
        self.capture_image()

    def draw(self, renderer, place, draw_pass, selected_only = False):
        if draw_pass == self.OPAQUE_DRAW_PASS:
            self.changed = self.capture_image()
        elif self.changed:
            Drawing.draw(self, renderer, place, draw_pass, selected_only)

    def capture_image(self):

        self.redraw_needed()

        # Capture current image
        v = self.viewer
        w,h = v.window_size
        rgba = v.frame_buffer_rgba()

        if self.rgba is None:
            self.rgba = rgba
            # Make textured square surface piece
            from .drawing import rgba_drawing
            self.piece = rgba_drawing(rgba, (-1,-1), (2,2), self)
            v.add_overlay(self)
        elif self.rgba.shape != (h,w,4):
            # Resize texture and motion blur image
            self.remove_drawing(self.piece)
            from .drawing import rgba_drawing
            self.piece = rgba_drawing(rgba, (-1,-1), (2,2), self)
            self.rgba = rgba
        else:
            from numpy import array
            bgcolor = array([255*c for c in v.background_color[:3]], rgba.dtype)
            alpha = 255*self.attenuate
            from .. import map_cpp
            c = map_cpp.blur_blend_images(self.decay_factor, rgba, self.rgba,
                                          bgcolor, alpha, self.rgba)
            if c == 0:
                return False    # No change
            self.piece.texture.reload_texture(self.rgba)
        self.redraw_needed()
        return True

    def delete(self):
        Drawing.delete(self)
