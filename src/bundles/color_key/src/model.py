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

_model = None
def get_model(session, *, create=True, add_created=True):
    if not _model and create:
        # constructor sets _model, which makes session restore easier
        ColorKeyModel(session)
        if add_created:
            session.models.add([_model], root_model = True)
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

    LC_FRACT = 4/5
    DEFAULT_SIZE = (0.25, 0.05)

    def __init__(self, session):
        super().__init__("Color key", session)
        global _model
        _model = self

        self._window_size = None
        self._texture_pixel_scale = 1
        self._aspect = 1
        self.needs_update = True
        self._small_key = False

        self.triggers.add_trigger("key changed")
        self.triggers.add_trigger("key closed")

        self._position = (0.7, 0.08)
        self._size = self.DEFAULT_SIZE
        self._rgbas_and_labels = [((0,0,1,1), "min"), ((1,1,1,1), ""), ((1,0,0,1), "max")]
        self._numeric_label_spacing = self.NLS_PROPORTIONAL
        self._color_treatment = self.CT_BLENDED
        self._justification = self.JUST_DECIMAL
        self._label_offset = 0
        self._label_side = self.LS_RIGHT_BOTTOM
        self._label_rgba = None
        self._bold = False
        self._italic = False
        self._font_size = 24
        self._font = 'Arial'
        self._border = True
        self._border_rgba = None
        self._border_width = 2
        self._ticks = False
        self._tick_length = 10
        self._tick_thickness = 4

        self._background_handler = None
        self._update_background_handler()

        session.main_view.add_overlay(self)

    def delete(self):
        if self._background_handler:
            from chimerax.core.core_settings import settings
            settings.triggers.remove_handler(self._background_handler)
        self.session.main_view.remove_overlays([self], delete = False)
        self.triggers.activate_trigger("key closed", None)
        super().delete()
        global _model
        _model = None

    def draw(self, renderer, draw_pass):
        self._update_graphics(renderer)
        if self._small_key:
            return
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
        self.triggers.activate_trigger("key changed", "bold")

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self, border):
        if border == self._border:
            return
        self._border = border
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "border")

    @property
    def border_rgba(self):
        # None means contrast with background
        return self._border_rgba

    @border_rgba.setter
    def border_rgba(self, rgba):
        from numpy import array_equal
        if array_equal(rgba, self._border_rgba):
            return
        self._border_rgba = rgba
        self.needs_update = True
        self.redraw_needed()
        self._update_background_handler()
        self.triggers.activate_trigger("key changed", "border_rgba")

    @property
    def border_width(self):
        return self._border_width

    @border_width.setter
    def border_width(self, border_width):
        if border_width == self._border_width:
            return
        self._border_width = border_width
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "border_width")

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
        self.triggers.activate_trigger("key changed", "color_treatment")

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
        self.triggers.activate_trigger("key changed", "font")

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
        self.triggers.activate_trigger("key changed", "font_size")

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
        self.triggers.activate_trigger("key changed", "italic")

    @property
    def justification(self):
        return self._justification

    @justification.setter
    def justification(self, just):
        if just == self.JUST_DECIMAL.split()[0]:
            just = self.JUST_DECIMAL
        if just == self._justification:
            return
        self._justification = just
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "justification")

    @property
    def label_offset(self):
        return self._label_offset

    @label_offset.setter
    def label_offset(self, offset):
        if offset == self._label_offset:
            return
        self._label_offset = offset
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "label_offset")

    @property
    def label_rgba(self):
        # None means contrast with background
        return self._label_rgba

    @label_rgba.setter
    def label_rgba(self, rgba):
        from numpy import array_equal
        if array_equal(rgba, self._label_rgba):
            return
        self._label_rgba = rgba
        self.needs_update = True
        self.redraw_needed()
        self._update_background_handler()
        self.triggers.activate_trigger("key changed", "label_rgba")

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
        self.triggers.activate_trigger("key changed", "label_side")

    @property
    def numeric_label_spacing(self):
        return self._numeric_label_spacing

    @numeric_label_spacing.setter
    def numeric_label_spacing(self, spacing):
        if spacing == self.NLS_PROPORTIONAL.split()[0]:
            spacing = self.NLS_PROPORTIONAL
        if spacing == self._numeric_label_spacing:
            return
        self._numeric_label_spacing = spacing
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "numeric_label_spacing")

    @property
    def pos(self):
        # was named 'position', but conflicts with Model.position
        return self._position

    @pos.setter
    def pos(self, llxy):
        # was named 'position', but conflicts with Model.position
        if llxy == self._position:
            return
        self._position = llxy
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "pos")

    @property
    def rgbas_and_labels(self):
        return self._rgbas_and_labels

    @rgbas_and_labels.setter
    def rgbas_and_labels(self, r_l):
        # skip the equality test since there are numpy arrays involved, sigh...
        self._rgbas_and_labels = r_l
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "rgbas_and_labels")

    @property
    def model_color(self):
        return False

    @model_color.setter
    def model_color(self, val):
        pass

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        if size == self._size:
            return
        self._size = size
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "size")

    @property
    def ticks(self):
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        if ticks == self._ticks:
            return
        self._ticks = ticks
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "ticks")

    @property
    def tick_length(self):
        return self._tick_length

    @tick_length.setter
    def tick_length(self, tick_length):
        if tick_length == self._tick_length:
            return
        self._tick_length = tick_length
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "tick_length")

    @property
    def tick_thickness(self):
        return self._tick_thickness

    @tick_thickness.setter
    def tick_thickness(self, tick_thickness):
        if tick_thickness == self._tick_thickness:
            return
        self._tick_thickness = tick_thickness
        self.needs_update = True
        self.redraw_needed()
        self.triggers.activate_trigger("key changed", "tick_thickness")

    session_attrs = [
            "_bold",
            "_border",
            "_border_rgba",
            "_border_width",
            "_color_treatment",
            "_font",
            "_font_size",
            "_italic",
            "_justification",
            "_label_offset",
            "_label_rgba",
            "_label_side",
            "_numeric_label_spacing",
            "_position",
            "_rgbas_and_labels",
            "_size",
            "_ticks",
            "_tick_length",
            "_tick_thickness",
    ]

    def take_snapshot(self, session, flags):
        data = { attr: getattr(self, attr) for attr in self.session_attrs }
        data['model state'] = Model.take_snapshot(self, session, flags)
        return data

    @staticmethod
    def restore_snapshot(session, data):
        key = get_model(session, add_created=False)
        Model.set_state_from_snapshot(key, session, data.pop('model state'))
        for attr, val in data.items():
            setattr(key, attr, val)
        key.needs_update = True
        key.redraw_needed()
        key._update_background_handler()
        session.models.add([key], root_model = True)
        return key

    def _update_background_handler(self):
        need_handler = self._label_rgba is None or self._border_rgba is None
        from chimerax.core.core_settings import settings
        if need_handler and not self._background_handler:
            def check_setting(trig_name, data, key=self):
                if data[0] == 'background_color':
                    key.needs_update = True
                    key.redraw_needed()
            self._background_handler = settings.triggers.add_handler('setting changed', check_setting)
        elif not need_handler and self._background_handler:
            settings.triggers.remove_handler(self._background_handler)
            self._background_handler = None

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
            sw, sh = self.session.main_view.window_size
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
        self._set_key_image(rgba)

    def _key_image_rgba(self):
        key_w, key_h = self._size
        win_w, win_h = self._window_size
        rect_pixels = [int(win_w * key_w + 0.5), int(win_h * key_h + 0.5)]
        pixels = rect_pixels[:]
        self.start_offset = start_offset = [0, 0]
        self.end_offset = end_offset = [0, 0]
        border = self._border_width if self._border else 0
        tick_length = self._tick_length if self._ticks else 0
        label_offset = self._label_offset + 5 + border + tick_length
        if pixels[0] > pixels[1]:
            layout = "horizontal"
            long_index = 0
        else:
            layout = "vertical"
            long_index = 1
        rgbas, labels = zip(*self._rgbas_and_labels)
        if rect_pixels[long_index] < len(rgbas):
            # Don't draw tiny keys
            self._small_key = True
            return None
        self._small_key = False
        rect_positions = self._rect_positions(pixels[long_index])

        if self._color_treatment == self.CT_BLENDED:
            label_positions = rect_positions
        else:
            label_positions = [(rect_positions[i] + rect_positions[i+1])/2
                for i in range(len(rect_positions)-1)]

        rgbas = [(int(255*r + 0.5), int(255*g + 0.5), int(255*b + 0.5), int(255*a + 0.5))
            for r,g,b,a in rgbas]

        if layout == "vertical":
            rgbas = list(reversed(rgbas))
            label_positions = list(reversed(label_positions))
            labels = list(reversed(labels))
            first_index, last_index = -1, 0
        else:
            first_index, last_index = 0, -1

        if border:
            for i in range(2):
                start_offset[i] += border
                end_offset[i] += border
                pixels[i] += 2 * border
        if tick_length:
            pixels[1-long_index] += tick_length
            if self._label_side == self.LS_LEFT_TOP:
                start_offset[1-long_index] += tick_length
            else:
                end_offset[1-long_index] += tick_length

        from Qt.QtGui import QImage, QPainter, QColor, QBrush, QPen, QLinearGradient, QFontMetrics, QFont
        from Qt.QtCore import Qt, QRectF, QPointF

        font = QFont(self.font, int(self.font_size * self._texture_pixel_scale), (QFont.Bold if self.bold else QFont.Normal), self.italic)
        fm = QFontMetrics(font)
        top_label_y_offset = font_height = fm.ascent()
        font_descender = fm.descent()
        # text is centered vertically from 0 to height (i.e. ignore descender) whereas it is centered
        # horizontally across the full width
        #
        # fm.boundingRect(label) basically returns a fixed height for all labels (and a large negative y)
        # so just use the font size instead
        if labels[first_index]:
            # may need extra room to left or bottom for first label
            bounds = fm.boundingRect(labels[first_index])
            xywh = bounds.getRect()
            # Qt seemingly will not return the actual height of a text string; estimate all lower case
            # to be LC_FRACT height
            label_height = (font_height * self.LC_FRACT) if labels[first_index].islower() else font_height
            label_size = label_height if layout == "vertical" else xywh[long_index+2]
            extra = max(label_size / 2 - label_positions[first_index] - border, 0)
            (end_offset if layout == "vertical" else start_offset)[long_index] += extra
            pixels[long_index] += extra

        # need room below or to right to layout labels
        # if layout is vertical and justification is decimal-point, the "widest label" is actually the
        # combination of the widest to the left of the decimal point + the widest to the right of it
        decimal_widths = []
        if layout == "vertical" and self._justification == self.JUST_DECIMAL:
            left_widest = right_widest = 0
            for label in labels:
                if label is None:
                    label = ""
                overall = fm.boundingRect(label).getRect()[2]
                if '.' in label:
                    right = fm.boundingRect(label[label.index('.'):]).getRect()[2]
                    left = overall - right
                else:
                    left = overall
                    right = 0
                left_widest = max(left_widest, left)
                right_widest = max(right_widest, right)
                decimal_widths.append((left, right))
            extra = left_widest + right_widest + label_offset
        else:
            if layout == "vertical":
                extra = max([fm.boundingRect(lab).getRect()[3-long_index] for lab in labels]) + label_offset
            else:
                # Qt seemingly will not return the actual height of a text string; estimate all lower case
                # to be LC_FRACT height
                for label in labels:
                    if label and not label.islower():
                        label_height = font_height
                        break
                else:
                    label_height = top_label_y_offset = font_height * self.LC_FRACT
                extra = label_height + label_offset
            decimal_widths = [(None, None)] * len(labels)
        if self._label_side == self.LS_LEFT_TOP:
            start_offset[1-long_index] += extra
        else:
            end_offset[1-long_index] += extra
        pixels[1-long_index] += extra

        pixels[1] += font_descender
        end_offset[1] += font_descender

        if labels[last_index]:
            # may need extra room to right or top for last label
            bounds = fm.boundingRect(labels[last_index])
            xywh = bounds.getRect()
            # Qt seemingly will not return the actual height of a text string; estimate all lower case
            # to be LC_FRACT height
            label_height = (font_height * self.LC_FRACT) if labels[last_index].islower() else font_height
            label_size = label_height if layout == "vertical" else xywh[long_index+2]
            extra = max(label_size / 2 - (rect_pixels[long_index] - label_positions[last_index]) - border, 0)
            (start_offset if layout == "vertical" else end_offset)[long_index] += extra
            pixels[long_index] += extra

        image = QImage(max(int(pixels[0]), 1), max(int(pixels[1]), 1), QImage.Format_ARGB32)
        image.fill(QColor(0,0,0,0))    # Set background transparent

        from chimerax.core.colors import contrast_with_background
        try:
            p = QPainter(image)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(Qt.NoPen))

            border_rgba = contrast_with_background(self.session) \
                if self._border_rgba is None else self._border_rgba
            border_qcolor = QColor(*[int(255.0*c + 0.5) for c in border_rgba])
            if border:
                border_rgba = contrast_with_background(self.session) \
                    if self._border_rgba is None else self._border_rgba
                p.setBrush(border_qcolor)
                p.setPen(QPen(border_qcolor, 0))
                # because of the way pen stroking works, subtract 0.5 from the "correct" y values
                adjustment = 0.0 if layout == "vertical" else -0.5
                corners = [QPointF(start_offset[0] - border, start_offset[1] - border + adjustment),
                    QPointF(pixels[0] - end_offset[0] + border,
                        pixels[1] - end_offset[1] + border + adjustment)]
                p.drawRect(QRectF(*corners))
            if self._color_treatment == self.CT_BLENDED:
                edge1, edge2 = start_offset[1-long_index], pixels[1-long_index] - end_offset[1-long_index]
                for i in range(len(rect_positions)-1):
                    start = rect_positions[i]
                    stop = rect_positions[i+1]
                    if layout == "vertical":
                        offset = end_offset[long_index]
                        start, stop = pixels[1] - stop - offset, pixels[1] - start - offset
                        x1, y1, x2, y2 = edge1, start, edge2, stop
                        gradient = QLinearGradient(0, start, 0, stop)
                        i1, i2 = len(rgbas)-i-2, len(rgbas)-i-1
                    else:
                        offset = start_offset[long_index]
                        x1, y1, x2, y2 = start + offset, edge1, stop + offset, edge2
                        gradient = QLinearGradient(x1, 0, x2, 0)
                        i1, i2 = i, i+1
                    gradient.setColorAt(0, QColor(*rgbas[i1]))
                    gradient.setColorAt(1, QColor(*rgbas[i2]))
                    p.setBrush(QBrush(gradient))
                    p.setPen(QPen(QBrush(gradient), 0))
                    p.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
                # The one-call gradient below doesn't seem to position the transition points
                # completely correctly, whereas the above piecemeal code does
                """
                gradient = QLinearGradient()
                gradient.setCoordinateMode(QLinearGradient.ObjectMode)
                rect_start = [start_offset[i] / pixels[i] for i in (0,1)]
                rect_end = [(pixels[i] - end_offset[i]) / pixels[i] for i in (0,1)]
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
                p.drawRect(QRectF(QPointF(start_offset[0], start_offset[1]),
                    QPointF(pixels[0] - end_offset[0], pixels[1] - end_offset[1])))
                """
            else:
                for i in range(len(rect_positions)-1):
                    if layout == "vertical":
                        x1, y1 = 0, pixels[1] - rect_positions[i]
                        x2, y2 = rect_pixels[0], pixels[1] - rect_positions[i+1]
                        y_offset = - end_offset[1]
                        rgba_index = len(rgbas) - i - 1
                    else:
                        x1, y1 = rect_positions[i], 0
                        x2, y2 = rect_positions[i+1], rect_pixels[1]
                        y_offset = start_offset[1]
                        rgba_index = i
                    brush = QBrush(QColor(*rgbas[rgba_index]))
                    p.setBrush(brush)
                    p.drawRect(QRectF(QPointF(x1 + start_offset[0], y1 + y_offset),
                        QPointF(x2 + start_offset[0], y2 + y_offset)))
            p.setFont(font)
            label_rgba = contrast_with_background(self.session) \
                if self._label_rgba is None else self._label_rgba
            for label, pos, decimal_info in zip(labels, label_positions, decimal_widths):
                if not label:
                    continue
                p.setPen(QColor(*[int(255.0*c + 0.5) for c in label_rgba]))
                rect = fm.boundingRect(label)
                if layout == "vertical":
                    if self._justification == self.JUST_DECIMAL:
                        pre_decimal_width, decimal_width = decimal_info
                        if self._label_side == self.LS_LEFT_TOP:
                            x = start_offset[0] - right_widest - pre_decimal_width - label_offset
                        else:
                            x = pixels[0] - right_widest - pre_decimal_width
                    elif self._justification == self.JUST_LEFT:
                        if self._label_side == self.LS_LEFT_TOP:
                            x = 0
                        else:
                            x = pixels[0] - end_offset[0] + label_offset
                    else:
                        if self._label_side == self.LS_LEFT_TOP:
                            x = start_offset[0] - (rect.width() - rect.x()) - label_offset
                        else:
                            x = pixels[0] - (rect.width() - rect.x())
                    # Qt seemingly will not return the actual height of a text string; estimate all
                    # lower case to be LC_FRACT height
                    label_height = (font_height * self.LC_FRACT) if label.islower() else font_height
                    y = pixels[1] - end_offset[1] - pos + label_height / 2
                else:
                    if self._label_side == self.LS_LEFT_TOP:
                        y = top_label_y_offset
                    else:
                        y = pixels[1] - font_descender
                    x = start_offset[0] + pos - (rect.width() - rect.x())/2
                p.drawText(int(x), int(y), label)

                if tick_length:
                    tick_thickness = self._tick_thickness
                    if layout == "vertical":
                        if self._label_side == self.LS_LEFT_TOP:
                            x1 = start_offset[0] - border - tick_length
                        else:
                            x1 = pixels[0] - end_offset[0] + border
                        # because of the way pen stroking works, subtract 0.5 from the "correct" y values
                        adjustment = 0.0 if layout == "vertical" else -0.5
                        y1 = pixels[1] - end_offset[1] - pos - tick_thickness/2 + adjustment
                        x2, y2 = x1 + tick_length, y1 + tick_thickness
                    else:
                        if self._label_side == self.LS_LEFT_TOP:
                            y1 = start_offset[1] - border - tick_length
                        else:
                            y1 = pixels[1] - end_offset[1] + border
                        x1 = start_offset[0] + pos - tick_thickness/2
                        x2, y2 = x1 + tick_thickness, y1 + tick_length
                    p.setPen(QPen(border_qcolor, 0))
                    p.setBrush(border_qcolor)
                    p.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
        finally:
            p.end()

        # Convert to numpy rgba array
        from chimerax.graphics import qimage_to_numpy
        return qimage_to_numpy(image)

    def _rect_positions(self, long_size):
        proportional = False
        texts = [color_text[1] for color_text in self._rgbas_and_labels]
        if self._numeric_label_spacing == self.NLS_PROPORTIONAL:
            import locale
            restore_locale = True
            try:
                local_numeric = locale.getlocale(locale.LC_NUMERIC)
            except locale.Error:
                restore_locale = False
            try:
                try:
                    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
                except locale.Error:
                    # hope current locale works!
                    pass
                try:
                    values = [locale.atof('' if t is None else t) for t in texts]
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
                    if proportional and values[0] == values[-1]:
                        proportional = False
            finally:
                if restore_locale:
                    locale.setlocale(locale.LC_NUMERIC, local_numeric)
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
                for i in range(len(values)-1)] + [long_size]
        return rect_positions

    def _set_key_image(self, rgba):
        if self._small_key:
            return
        key_x, key_y = self._position
        # adjust to the corner of the key itself, excluding labels etc.
        w, h = self._window_size
        key_x -= self.start_offset[0] / w
        key_y -= self.end_offset[1] / h
        x, y = (-1 + 2*key_x, -1 + 2*key_y)    # Convert 0-1 position to -1 to 1.
        y *= self._aspect
        th, tw = rgba.shape[:2]
        uw, uh = 2*tw/w, 2*th/h
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, rgba, (x, y), (uw, uh), opaque=False)
