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
    color_treatments = (CT_BLENDED, CT_DISTINCT)

    JUST_LEFT = "left"
    JUST_DECIMAL = "decimal point"
    JUST_RIGHT = "right"
    justifications = (JUST_LEFT, JUST_DECIMAL, JUST_RIGHT)

    LS_LEFT_TOP = "left/top"
    LS_RIGHT_BOTTOM = "right/bottom"
    label_sides = (LS_LEFT_TOP, LS_RIGHT_BOTTOM)

    NLS_EQUAL = "equal"
    NLS_PROPORTIONAL = "proportional to value"
    numeric_label_spacings = (NLS_EQUAL, NLS_PROPORTIONAL)


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
        #TODO: actually fire these appropriately
        self.key_triggers.add_trigger("changed")
        self.key_triggers.add_trigger("closed")

        self._position = None
        self._size = (0.25, 0.05)
        self._rgbas_and_labels = [((0,0,1,1), "min"), ((1,1,1,1), ""), ((1,0,0,1), "max")]
        self._numeric_label_spacing = self.NLS_PROPORTIONAL
        self._color_treatment = self.CT_BLENDED
        self._justification = JUST_DECIMAL
        self._label_size = LS_RIGHT_BOTTOM
        self._bold = False
        self._italic = False
        self._font_size = 24
        self._font = 'Arial'

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
    def bold(self):
        return self._bold

    @bold.setter
    def bold(self, bold):
        if bold == self._bold:
            return
        self._bold = bold
        self.needs_update = True
        self.redraw_needed()

    @property
    def color_treatment(self):
        return self._color_treatment

    @color_treatment.setter
    def color_treatment(self, ct):
        if ct == self._color_treatment:
            return
        self._color_treatment = ct
        self.needs_update = True
        self.redraw_needed()

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, font):
        if font == self._font:
            return
        self._font = font
        self.needs_update = True
        self.redraw_needed()

    @property
    def font_size(self):
        return self._font_size

    @font_size.setter
    def font_size(self, font_size):
        if font_size == self._font_size:
            return
        self._font_size = font_size
        self.needs_update = True
        self.redraw_needed()

    @property
    def italic(self):
        return self._italic

    @italic.setter
    def italic(self, italic):
        if italic == self._italic:
            return
        self._italic = italic
        self.needs_update = True
        self.redraw_needed()

    @property
    def justification(self):
        return self._justification

    @justification.setter
    def justification(self, just):
        if just == self._justification:
            return
        self._justification = just
        self.needs_update = True
        self.redraw_needed()

    @property
    def label_side(self):
        return self._label_side

    @label_side.setter
    def label_side(self, side):
        if side == self._label_side:
            return
        self._label_side = side
        self.needs_update = True
        self.redraw_needed()

    @property
    def numeric_label_spacing(self):
        return self._numeric_label_spacing

    @numeric_label_spacing.setter
    def numeric_label_spacing(self, spacing):
        if spacing == self._numeric_label_spacing:
            return
        self._numeric_label_spacing = spacing
        self.needs_update = True
        self.redraw_needed()

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
        rect_pixels = [int(win_w * key_w + 0.5), int(win_h * key_h + 0.5)]
        pixels = rect_pixels[:]
        start_offset = [0, 0]
        end_offset = [0, 0]
        if pixels[0] > pixels[1]:
            layout = "horizontal"
            long_index = 0
        else:
            layout = "vertical"
            long_index = 1
        rect_positions = self._rect_positions(pixels[long_index])

        if self._color_treatment == self.CT_BLENDED:
            label_positions = rect_positions
        else:
            label_positions = [(rect_positions[i] + rect_positions[i+1])/2
                for i in range(len(rect_positions)-1)]

        rgbas, labels = zip(*self._rgbas_and_labels)
        rgbas = [(int(255*r + 0.5), int(255*g + 0.5), int(255*b + 0.5), int(255*a + 0.5))
            for r,g,b,a in rgbas]

        from PySide2.QtGui import QImage, QPainter, QColor, QBrush, QPen, QLinearGradient, QFontMetrics, \
            QFont
        from PySide2.QtCore import Qt, QRect, QPoint

        font = QFont(self.font, self.font_size, (QFont.Bold if self.bold else QFont.Normal), self.italic)
        fm = QFontMetrics(font)
        # text is centered vertically from 0 to height (i.e. ignore descender) whereas it is centered
        # horizontally across the full width
        if labels[0]:
            # may need extra room to left or top for first label
            bounds = fm.boundingRect(labels[0])
            xywh = bounds.getRect()
            label_size = xywh[long_index+2] + (xywh[1] if layout == "vertical" else 0)
            extra = max(label_size / 2 - label_positions[0][long_index], 0)
            start_offset[long_index] += extra
            pixels[long_index] += extra

        # need room below or to right to layout labels
        # if layout is vertical and justification is decimal-point, the "widest label" is actually the
        # combination of the widest to the left of the decimal point + the widest to the right of it
        decimal_widths = []
        if layout == "vertical" and self._justification == self.JUST_DECIMAL:
            left_widest = right_widest = 0
            for label in labels:
                if '.' in label:
                    left = fm.boundingRect(label[:label.index('.')]).getRect()[2]
                    right = fm.boundingRect(label[label.index('.'):]).getRect()[2]
                else:
                    left = fm.boundingRect(label).getRect()[2]
                    right = 0
                left_widest = max(left_widest, left)
                right_widest = max(right_widest, right)
                decimal_widths.append((left, right))
            extra = left_widest + right_widest
        else:
            extra = max([fm.boundingRect(lab).getRect()[3-long_index] for lab in labels])
        extra = max([fm.boundingRect(lab).getRect()[3-long_index] for lab in labels])
        if self._labels_side == self.LS_LEFT_TOP:
            start_offset[1-long_index] += extra
        else:
            end_offset[1-long_index] += extra
        pixels[1-long_index] += extra

        if labels[-1]:
            # may need extra room to right or bottom for last label
            bounds = fm.boundingRect(labels[0])
            xywh = bounds.getRect()
            # since vertical labels are centered on the half height ignoring the descender,
            # need to use the full height here (which includes the descender) *plus*
            # the descender again, in order to account for the full descender after halving
            label_size = xywh[long_index+2] - (xywh[1] if layout == "vertical" else 0)
            extra = max(label_size / 2 - (rect_pixels[long_index] - label_positions[-1][long_index]), 0)
            end_offset[long_index] += extra
            pixels[long_index] += extra

        image = QImage(max(pixels[0], 1), max(pixels[0], 1), QImage.Format_ARGB32)
        image.fill(QColor(0,0,0,0))    # Set background transparent

        try:
            p = QPainter(image)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(Qt.NoPen))

            if self._color_treatment == self.CT_BLENDED:
                gradient = QLinearGradient()
                gradient.setCoordinateMode(QLinearGradient.ObjectMode)
                rect_start = [start_offset[i] / pixels[i] for in in (0,1)]
                rect_end = [(pixels[i] - end_offset[i]) / x_pixels[i] for i in (0,1)]
                if layout == "vertical":
                    start, stop = (rect_start[0], rect_end[1]), (rect_start[0], rect_start[1])
                else:
                    start, stop = (rect_start[0], rect_start[1]), (rect_end[0], rect_start[1])
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
                        x2, y2 = rect_pixels[0], rect_positions[i+1]
                    else:
                        x1, y1 = rect_positions[i], 0
                        x2, y2 = rect_positions[i+1], rect_pixels[1]
                    p.drawRect(QRect(QPoint(x1 + start_offset[0], y1 + start_offset[1]),
                        QPoint(x2 + start_offset[0], y2 + start_offset[1])))
            p.setFont(font)
            #TODO: draw labels (incl. justification)
        finally:
            p.end()

        # Convert to numpy rgba array
        from chimerax.graphics import qimage_to_numpy
        return qimage_to_numpy(image)

    def _rect_positions(self, long_size):
        proportional = False
        texts = [color_text[1] for color_text in self._rgbas_and_labels]
        if self._numeric_label_spacing == self.NLS_PROPORTIONAL:
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
