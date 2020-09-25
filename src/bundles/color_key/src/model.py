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

_model = None
def get_model(session, *, create=True):
    if not _model and create:
        # constructor sets _model, which makes session restore easier
        ColorKeyModel(session)
    return _model

from chimerax.core.models import Model

class ColorKeyModel(Model):

    pickable = False
    casts_shadows = False

    CT_BLENDED = "blended"
    CT_DISTINCT = "distinct"
    NLS_EQUAL = "equal"
    NLS_PROPORTIONAL = "proportional to value"

    def __init__(self, session):
        super().__init__("Color key", session)
        global _model
        _model = self

        self._window_size = None
        self._texture_pixel_scale = 1
        self._aspect = 1
        self.needs_update = True

        from chimerax.core.triggerset import TriggerSet
        self.key_triggers = TriggerSet()
        self.key_triggers.add_trigger("changed")
        self.key_triggers.add_trigger("closed")

        self._position = None
        self._size = (0.25, 0.05)
        self._rgbas_and_labels = [((0,0,1,1), "min"), ((1,1,1,1), ""), ((1,0,0,1), "max")]
        self._num_label_spacing = self.NLS_PROPORTIONAL
        self._color_treatment = self.CT_BLENDED

        session.main_view.add_overlay(self)

    def delete(self):
        self.session.main_view.remove_overlays([self], delete = False)
        self.key_triggers.activate_trigger("closed", None)
        super().delete()
        global _model
        _model = None

    def draw(self, renderer, draw_pass):
        if self._position is None:
            return
        self._update_graphics(renderer)
        super().draw(renderer, draw_pass)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, llxy):
        if llxy == self._position:
            return
        self._position = llxy
        self.needs_update = True
        self.redraw_needed()

    @property
    def rgbas_and_labels(self):
        return self._rgbas_and_labels

    @rgbas_and_labels.setter
    def rgbas_and_labels(self, r_l):
        # skip the equality test since there are numpy arrays involved, sigh...
        self._rgbas_and_labels = r_l
        self.needs_update = True
        self.redraw_needed()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, wh):
        if wh == self._size:
            return
        self._size = wh
        self.needs_update = True
        self.redraw_needed()

    def _update_graphics(self, renderer):
        """
        Recompute the key's texture image or update its texture coordinates that position it in the window
        based on the rendered window size.  When saving an image file, the rendered size may differ from the
        on-screen window size.  In that case, make the label size match its relative size seen on the screen.
        """
        window_size = renderer.render_size()

        # Remember the on-screen size if rendering off screen
        if getattr(renderer, 'image_save', False):
            # When saving an image, match the key's on-screen fractional size, even though the image size
            # in pixels may be different from on screen
            sw, sh = self.session.main_window.window_size
            w, h = window_size
            pixel_scale = (w / sw) if sw > 0 else 1
            aspect = (w*sh) / (h*sw) if h*sw > 0 else 1
        else:
            pixel_scale = renderer.pixel_scale()
            aspect = 1
        if pixel_scale != self._texture_pixel_scale or aspect != self._aspect:
            self._texture_pixel_scale = pixel_scale
            self._aspect = aspect
            self.needs_update = True

        # Need to reposition key if window size changes
        win_size_changed = (window_size != self._window_size)
        if win_size_changed:
            self._window_size = window_size
            self.needs_update = True

        if self.needs_update:
            self.needs_update = False
            self._update_key_image()

    def _update_key_image(self):
        rgba = self._key_image_rgba()
        if rgba is None:
            self.session.logger.info("Can't find font for color key labels")
        else:
            self._set_key_image(rgba)

    def _key_image_rgba(self):
        ulx, uly = self._position
        key_w, key_h = self._size
        win_w, win_h = self._window_size
        x_pixels = int(win_w * key_w + 0.5)
        y_pixels = int(win_h * key_h + 0.5)
        if x_pixels > y_pixels:
            layout = "horizontal"
            long_size = x_pixels
        else:
            layout = "vertical"
            long_size = y_pixels
        rect_positions = self._rect_positions(long_size)

        if self._color_treatment == self.CT_BLENDED:
            label_positions = rect_positions
        else:
            label_positions = [(rect_positions[i] + rect_positions[i+1])/2
                for i in range(len(rect_positions)-1)]

        from PyQt5.QtGui import QImage, QPainter, QColor, QBrush, QPen, QLinearGradient
        from PyQt5.QtCore import Qt, QRect, QPoint
        image = QImage(max(x_pixels, 1), max(y_pixels, 1), QImage.Format_ARGB32)
        image.fill(QColor(0,0,0,0))    # Set background transparent

        rgbas, labels = zip(*self._rgbas_and_labels)
        rgbas = [(int(255*r + 0.5), int(255*g + 0.5), int(255*b + 0.5), int(255*a + 0.5))
            for r,g,b,a in rgbas]

        with QPainter(image) as p:
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(Qt.NoPen))

            if self._color_treatment == self.CT_BLENDED:
                gradient = QLinearGradient()
                gradient.setCoordinateMode(QLinearGradient.ObjectMode)
                if layout == "vertical":
                    start, stop = (0.0, 1.0), (0.0, 0.0)
                else:
                    start, stop = (0.0, 0.0), (1.0, 0.0)
                gradient.setStart(*start)
                gradient.setFinalStop(*stop)
                for rgba, rect_pos in zip(rgbas, rect_positions):
                    fraction = rect_pos/rect_positions[-1]
                    if layout == "vertical":
                        fraction = 1.0 - fraction
                    gradient.setColorAt(fraction, QColor(*rgba))
                p.setBrush(QBrush(gradient))
                p.drawRect(QRect(QPoint(0, 0), QPoint(x_pixels, y_pixels)))
            else:
                for i in range(len(rect_positions)-1):
                    brush = QBrush(QColor(*rgbas[i]))
                    p.setBrush(brush)
                    if layout == "vertical":
                        x1, y1 = 0, rect_positions[i]
                        x2, y2 = x_pixels, rect_positions[i+1]
                    else:
                        x1, y1 = rect_positions[i], 0
                        x2, y2 = rect_positions[i+1], y_pixels
                    p.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

        # Convert to numpy rgba array
        from chimerax.graphics import qimage_to_numpy
        return qimage_to_numpy(image)

    def _rect_positions(self, long_size):
        proportional = False
        texts = [color_text[1] for color_text in self._rgbas_and_labels]
        if self._num_label_spacing == "proportional to value":
            try:
                values = [float(t) for t in texts]
            except (ValueError, TypeError):
                pass
            else:
                if values == sorted(values):
                    proportional = True
                else:
                    values.reverse()
                    if values == sorted(values):
                        proportional = True
                    values.reverse()
        if not proportional:
            values = range(len(texts))
        if self._color_treatment == self.CT_BLENDED:
            val_size = abs(values[0] - values[-1])
            rect_positions = [long_size * abs(v - values[0])/val_size for v in values]
        else:
            v0 = values[0] - (values[1] - values[0])/2.0
            vN = values[-1] + (values[-1] - values[-2])/2.0
            val_size = abs(vN-v0)
            positions = [long_size * abs(v-v0) / val_size for v in values]
            rect_positions = [0.0] + [(positions[i] + positions[i+1])/2.0
                for i in range(len(values)-1)] + [1.0]
        return rect_positions

    def _set_key_image(self, rgba):
        if self._position is None:
            return
        key_x, key_y = self._position
        x, y = (-1 + 2*key_x, -1 + 2*key_y)    # Convert 0-1 position to -1 to 1.
        y *= self._aspect
        w, h = self._window_size
        th, tw = rgba.shape[:2]
        self._texture_size = (tw, th)
        uw, uh = 2*tw/w, 2*th/h
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, rgba, (x, y), (uw, uh), opaque=False)
