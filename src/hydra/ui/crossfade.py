# Allow fading from one scene to another

from ..surface import Surface
class Cross_Fade(Surface):
    
    def __init__(self, viewer, frames):

        Surface.__init__(self, 'cross fade')
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
        w,h = v.window_size
        from ..draw import drawing
        self.rgba = drawing.frame_buffer_image_rgba8(w, h)

        # Make textured square surface piece
        from .. import surface
        self.piece = surface.rgba_surface_piece(self.rgba, (-1,-1), (2,2), self)

        v.add_overlay(self)
        v.add_new_frame_callback(self.next_frame)

    def next_frame(self):

        f,n = self.frame+1, self.frames
        if f >= n:
            v = self.viewer
            v.remove_new_frame_callback(self.next_frame)
            v.remove_overlays([self])
            self.removeAllPieces()
            return

        # Increase texture transparency
        self.frame = f
        alpha = int(255 * (n-f) / n)
        self.rgba[:,:,3] = alpha
        from ..draw import drawing
        drawing.reload_texture(self.piece.textureId, self.rgba)
        self.redraw_needed = True
