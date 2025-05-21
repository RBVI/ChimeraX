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

from chimerax.core.models import Model


def has_graphics(session):
    if session.main_view.renderer is None:
        from chimerax.core.errors import LimitationError

        raise LimitationError(
            "Unable to do list fonts without being able to render images"
        )
    return True


def label2d(
    session,
    labels=None,
    *,
    text=None,
    color=None,
    bg_color=None,
    size=None,
    font=None,
    bold=None,
    italic=None,
    xpos=None,
    ypos=None,
    visibility=None,
    margin=None,
    outline=None,
    frames=None
):
    keywords = (
        "text",
        "color",
        "bg_color",
        "size",
        "font",
        "bold",
        "italic",
        "xpos",
        "ypos",
        "visibility",
        "margin",
        "outline",
    )
    loc = locals()
    kw = {attr: loc[attr] for attr in keywords if loc[attr] is not None}
    if labels is None:
        if text is None:
            labels = all_labels(session)
        else:
            return label_create(session, name="", **kw)

    kw["frames"] = frames
    return [_update_label(session, label, **kw) for label in labels]


def label_create(
    session,
    name,
    text="",
    color=None,
    bg_color=None,
    size=24,
    font="Arial",
    bold=None,
    italic=None,
    xpos=0.5,
    ypos=0.5,
    visibility=True,
    margin=0,
    outline=0,
):
    """Create a label at a fixed position in the graphics window.

    Parameters
    ----------
    name : string
      Identifier for the label used to change or delete label.
    text : string
      Displayed text of the label.
    color : Color
      Color of the label text.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    bg_color : Color
      Draw rectangular label background in this color.  If omitted, background is transparent.
    size : int
      Font size in points.
    font : string
      Font name.  This must be a true type font installed on Mac in /Library/Fonts
      and is the name of the font file without the ".ttf" suffix.
    xpos : float
      Placement of left edge of text. Range 0 - 1 covers full width of graphics window.
    ypos : float
      Placement of bottom edge of text. Range 0 - 1 covers full height of graphics window.
    visibility : bool
      Whether or not to display the label.
    margin : float
      Amount of padding to add around text
    outline : float
      width of contrasting outline to place around background/margin
    """
    if name == "all":
        from chimerax.core.errors import UserError

        raise UserError("'all' is reserved to refer to all labels")
    elif name:
        lm = session_labels(session)
        if lm and lm.named_label(name) is not None:
            from chimerax.core.errors import UserError

            raise UserError('Label "%s" already exists' % name)

    kw = {
        "text": text,
        "color": color,
        "size": size,
        "font": font,
        "bold": bold,
        "italic": italic,
        "xpos": xpos,
        "ypos": ypos,
        "visibility": visibility,
        "margin": margin,
        "outline_width": outline,
    }

    from chimerax.core.colors import Color

    if isinstance(color, Color):
        kw["color"] = color.uint8x4()
    elif isinstance(color, str) and color in ("default", "auto"):
        kw["color"] = None

    if isinstance(bg_color, Color):
        kw["background"] = bg_color.uint8x4()
    elif isinstance(bg_color, str) and bg_color == "none":
        kw["background"] = None

    has_graphics = session.main_view.render is not None
    if not has_graphics(session):
        from chimerax.core.errors import LimitationError

        raise LimitationError("Unable to draw 2D labels without rendering images")

    return Label(session, name, **kw)


def label_change(
    session,
    labels,
    *,
    text=None,
    color=None,
    bg_color=None,
    size=None,
    font=None,
    bold=None,
    italic=None,
    xpos=None,
    ypos=None,
    visibility=None,
    margin=None,
    outline=None,
    frames=None
):
    """Change label parameters."""
    kw = {
        "text": text,
        "color": color,
        "bg_color": bg_color,
        "size": size,
        "font": font,
        "bold": bold,
        "italic": italic,
        "xpos": xpos,
        "ypos": ypos,
        "visibility": visibility,
        "margin": margin,
        "outline": outline,
        "frames": frames,
    }
    return [_update_label(session, label, **kw) for label in labels]


def _update_label(
    session,
    label,
    *,
    text=None,
    color=None,
    bg_color=None,
    size=None,
    font=None,
    bold=None,
    italic=None,
    xpos=None,
    ypos=None,
    visibility=None,
    margin=None,
    outline=None,
    frames=None
):
    if font is not None:
        label.font = font
    if bold is not None:
        label.bold = bold
    if text is not None:
        label.text = text
    if italic is not None:
        label.italic = italic
    if frames is None:
        if color is not None:
            label.color = None if color in ("default", "auto") else color.uint8x4()
        if bg_color is not None:
            label.background = None if bg_color == "none" else bg_color.uint8x4()
        if size is not None:
            label.size = size
        if xpos is not None:
            label.xpos = xpos
        if ypos is not None:
            label.ypos = ypos
        if visibility is not None:
            label.visibility = visibility
        if margin is not None:
            label.margin = margin
        if outline is not None:
            label.outline_width = outline
        label.update_drawing()
    else:
        _InterpolateLabel(
            session,
            label,
            color,
            bg_color,
            size,
            xpos,
            ypos,
            visibility,
            margin,
            outline,
            frames,
        )


class _InterpolateLabel:
    def __init__(
        self,
        session,
        label,
        color,
        bg_color,
        size,
        xpos,
        ypos,
        visibility,
        margin,
        outline_width,
        frames,
    ):
        self.label = label
        from numpy import array_equal

        # even if color/background not changing, need color1/2 and bg1/2 for visibility changes
        from numpy import array, uint8

        self.orig_color1 = None if label.color is None else label.color.copy()
        self.color1, self.color2 = array(label.drawing.label_color, dtype=uint8), (
            color.uint8x4() if color else color
        )
        self.bg1, self.bg2 = (
            None if label.background is None else label.background.copy()
        ), bg_color
        if color is None:
            # no change
            self.interp_color = False
        else:
            color2 = None if color == "none" else color.uint8x4()
            if array_equal(label.color, color2):
                self.interp_color = False
            else:
                self.interp_color = True
        if bg_color is None:
            # no change
            self.interp_background = False
        else:
            bg2 = None if bg_color == "none" else bg_color.uint8x4()
            if array_equal(label.background, bg2):
                self.interp_background = False
            elif label.background is None or bg2 is None:
                # abrupt transition if adding/losing background
                label.background = bg2
                self.interp_background = False
            else:
                self.bg1 = label.background
                self.bg2 = bg2
                self.interp_background = True
        self.size1, self.size2 = label.size, size
        self.xpos1, self.xpos2 = label.xpos, xpos
        self.ypos1, self.ypos2 = label.ypos, ypos
        self.visibility1, self.visibility2 = label.visibility, visibility
        if visibility is not None and self.visibility1 != self.visibility2:
            if self.label.color is None:
                # need to interpolate alpha, so set it to a real color;
                # the last frame will set it to the right final value
                self.label.color = array(self.label.drawing.label_color, dtype=uint8)
        self.margin1, self.margin2 = label.margin, margin
        self.outline_width1, self.outline_width2 = label.outline_width, outline_width
        self.frames = frames
        from chimerax.core.commands import motion

        motion.CallForNFrames(self.frame_cb, frames, session)

    def frame_cb(self, session, frame):
        fraction = frame / (self.frames - 1)
        if self.interp_color:
            if frame == self.frames - 1:
                self.label.color = self.color2
            else:
                self.label.color = (
                    (1 - fraction) * self.color1 + fraction * self.color2
                ).astype(self.color1.dtype)
        if self.interp_background:
            if frame == self.frames - 1:
                self.label.background = self.bg2
            else:
                self.label.background = (
                    (1 - fraction) * self.bg1 + fraction * self.bg2
                ).astype(self.bg1.dtype)
        if self.size2 is not None and self.size1 != self.size2:
            if frame == self.frames - 1:
                self.label.size = self.size2
            else:
                self.label.size = (1 - fraction) * self.size1 + fraction * self.size2
        if self.xpos2 is not None and self.xpos1 != self.xpos2:
            if frame == self.frames - 1:
                self.label.xpos = self.xpos2
            else:
                self.label.xpos = (1 - fraction) * self.xpos1 + fraction * self.xpos2
        if self.ypos2 is not None and self.ypos1 != self.ypos2:
            if frame == self.frames - 1:
                self.label.ypos = self.ypos2
            else:
                self.label.ypos = (1 - fraction) * self.ypos1 + fraction * self.ypos2
        if self.visibility2 is not None and self.visibility1 != self.visibility2:
            if frame == self.frames - 1:
                self.label.visibility = self.visibility2
                self.label.color = (
                    self.orig_color1 if self.color2 is None else self.color2
                )
                self.label.background = self.bg1 if self.bg2 is None else self.bg2
            else:
                # fake gradual change in visibility via alpha channel
                if self.visibility2:
                    # becoming shown
                    self.label.color[-1] = self.label.color.dtype.type(
                        self.color1[-1] * fraction
                    )
                    if self.label.background is not None:
                        self.label.background[-1] = self.label.background.dtype.type(
                            self.bg1[-1] * fraction
                        )
                else:
                    # becoming hidden
                    self.label.color[-1] = self.label.color.dtype.type(
                        self.color1[-1] * (1 - fraction)
                    )
                    if self.label.background is not None:
                        self.label.background[-1] = self.label.background.dtype.type(
                            self.bg1[-1] * (1 - fraction)
                        )
                self.label.visibility = True
        if self.margin2 is not None and self.margin1 != self.margin2:
            if frame == self.frames - 1:
                self.label.margin = self.margin2
            else:
                self.label.margin = (
                    1 - fraction
                ) * self.margin1 + fraction * self.margin2
        if (
            self.outline_width2 is not None
            and self.outline_width1 != self.outline_width2
        ):
            if frame == self.frames - 1:
                self.label.outline_width = self.outline_width2
            else:
                self.label.outline_width = (
                    1 - fraction
                ) * self.outline_width1 + fraction * self.outline_width2
        self.label.update_drawing()


def label_under_window_position(session, view, win_x, win_y):
    w, h = view.window_size
    fx = (win_x + 0.5) / w
    fy = 1 - (win_y + 0.5) / h  # win_y is 0 at top
    lm = session_labels(session)
    if lm is None:
        return None
    for lbl in lm.all_labels:
        dx, dy = fx - lbl.xpos, fy - lbl.ypos
        d = lbl.drawing
        if d.display and d.parents_displayed:
            lw, lh = d.size
            if dx >= 0 and dx < lw and dy >= 0 and dy < lh:
                return lbl
    return None


def label_delete(session, labels=None):
    """Delete label."""
    if labels is None or labels == "all":
        lm = session_labels(session, create=False)
        labels = lm.all_labels if lm else []
        from .arrows import session_arrows

        am = session_arrows(session, create=False)
        labels = labels + (am.all_arrows if am else [])
    for label in tuple(labels):
        label.delete()


def label_listfonts(session):
    """Report available fonts."""
    from Qt.QtGui import QFontDatabase

    fnames = list(QFontDatabase.families())
    fnames.sort()
    session.logger.info("%d fonts available:\n%s" % (len(fnames), "\n".join(fnames)))


class Labels(Model):
    def __init__(self, session, view):
        Model.__init__(self, "2D labels", session)
        self._labels = []
        self.view = view
        self._named_labels = {}  # Map label name to Label object
        from chimerax.core.core_settings import settings

        self.handler = settings.triggers.add_handler(
            "setting changed", self._background_color_changed
        )
        self.view.add_overlay(self)
        self.model_panel_show_expanded = False

    def delete(self):
        self.view.remove_overlays([self], delete=False)
        self.handler.remove()
        Model.delete(self)

    def add_label(self, label):
        self._labels.append(label)
        n = label.name
        if n:
            nl = self._named_labels.get(n)
            if nl:
                self._labels.remove(nl)
                nl.delete()
            self._named_labels[n] = label
        self.add([label.drawing])

    @property
    def all_labels(self):
        return self._labels

    def named_label(self, name):
        return self._named_labels.get(name)

    def label_names(self):
        return tuple(self._named_labels.keys())

    def delete_label(self, label):
        if label.name:
            del self._named_labels[label.name]
        try:
            self._labels.remove(label)
        except ValueError:
            pass
        if len(self._labels) == 0:
            self.delete()

    def _background_color_changed(self, tname, tdata):
        # Update label color when graphics background color changes.
        if tdata[0] == "background_color":
            for label in self.all_labels:
                if label.background is None:
                    label.update_drawing()

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {"model state": Model.take_snapshot(self, session, flags), "version": 4}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        if "model state" in data:
            s = Labels(session)
            Model.set_state_from_snapshot(s, session, data["model state"])
            session.models.add([s], root_model=True)
            if "labels state" in data:
                # Older sessions restored the labels here.
                s.set_state_from_snapshot(session, data)
        else:
            # Restoring old session before 2d labels were Models.
            # Need to create labels model after all models created
            # so it does not use a model id already in use.
            def restore_old_labels(trigger_name, session, data=data):
                s = session_labels(session, create=True)
                s.set_state_from_snapshot(session, data)
                return "delete handler"

            session.triggers.add_handler("end restore session", restore_old_labels)
            s = None
        return s

    def set_state_from_snapshot(self, session, data):
        self._labels = [Label(session, **ls) for ls in data["labels state"]]


def find_label(session, name):
    lm = session_labels(session)
    return lm.named_label(name) if lm else None


def all_labels(session):
    lm = session_labels(session)
    return lm.all_labels if lm else []


def session_labels(session, view, create=False):
    lmlist = session.models.list(type=Labels)
    if lmlist:
        lm = lmlist[0]
    elif create:
        lm = Labels(session, view)
        session.models.add([lm], root_model=True)
    else:
        lm = None
    return lm


class Label:
    def __init__(
        self,
        session,
        view,
        name,
        text="",
        color=None,
        background=None,
        size=24,
        font="Arial",
        bold=False,
        italic=False,
        xpos=0.5,
        ypos=0.5,
        visibility=True,
        margin=0,
        outline_width=0,
        scalebar_width=None,
        scalebar_height=None,
    ):
        self.session = session
        self.name = name
        self.text = text
        self.color = color
        self.view = view
        from chimerax.core.colors import Color

        if isinstance(background, Color):
            self.background = background.uint8x4()
        else:
            # may already be numpy array if being restored from a session
            self.background = background
        # Logical pixels.  On Mac retina display will render 2x more pixels.
        self.size = size
        self.font = font
        self.bold = bold
        self.italic = italic
        self.xpos = xpos
        self.ypos = ypos
        self.visibility = visibility
        self.margin = margin  # Logical pixels.
        self.outline_width = outline_width
        self.scalebar_width = scalebar_width  # Angstroms
        self.scalebar_height = scalebar_height  # Pixels
        self.drawing = LabelModel(session, self)
        self.drawing.display = visibility
        self.labels = Labels(session, view)
        self.labels.add_label(self)

    def update_drawing(self):
        d = self.drawing
        d.needs_update = True
        # Used to be in LabelModel.update_drawing(), but that doesn't get called if display is False!
        d.display = self.visibility
        d.redraw_needed()

    def delete(self):
        d = self.drawing
        if d is None:
            return
        self.drawing = None
        if not d.deleted:
            d.delete()
        lm = session_labels(self.session, self.view)
        if lm:
            lm.delete_label(self)

    @property
    def is_scalebar(self):
        return self.scalebar_width is not None


class LabelModel(Model):
    pickable = False
    casts_shadows = False

    def __init__(self, session, label):
        name = label.name if label.name else label.text
        Model.__init__(self, name, session)
        self.label = label
        self._window_size = None  # Full window size in render pixels
        self._texture_size = None  # Label image size in render pixels
        # Converts label.size from logical pixels to render pixels
        self._texture_pixel_scale = 1
        self._aspect = (
            1  # Scale y label positioning for image saving at non-screen aspect ratio
        )
        self.needs_update = True

    def delete(self):
        Model.delete(self)
        self.label.delete()

    def draw(self, renderer, draw_pass):
        self._update_graphics(renderer)
        Model.draw(self, renderer, draw_pass)

    def _update_graphics(self, renderer):
        """
        Recompute the label texture image or update its texture coordinates that
        position it in the window based on the rendered window size.
        When saving an image file the rendered size may differ from the on screen
        window size.  In that case make the label size match its relative size
        seen on screen.
        """
        window_size = renderer.render_size()

        # Preserve on screen label size if saving an image of different size.
        if getattr(renderer, "image_save", False):
            # When saving an image match the label's on screen fractional size
            # even though the image size in pixels may be different from on screen.
            sw, sh = self.view.window_size
            w, h = window_size
            pscale = (w / sw) if sw > 0 else 1
            aspect = (w * sh) / (h * sw) if h * sw > 0 else 1
        else:
            pscale = renderer.pixel_scale()
            aspect = 1
        if pscale != self._texture_pixel_scale or aspect != self._aspect:
            self._texture_pixel_scale = pscale
            self._aspect = aspect
            self.needs_update = True

        # Will need to reposition label if window size changes.
        win_size_changed = window_size != self._window_size
        if win_size_changed:
            self._window_size = window_size

        if self.needs_update:
            self.needs_update = False
            self._update_label_image()
        elif win_size_changed:
            self._position_label_image()
        elif self.label.is_scalebar:
            self._position_label_image()

    def _update_label_image(self):
        label = self.label
        if not label.is_scalebar:
            xpad = (
                0 if label.background is None else int(0.2 * label.size)
            ) + label.margin
            ypad = label.margin
            s = self._texture_pixel_scale
            from chimerax.graphics import text_image_rgba

            rgba = text_image_rgba(
                label.text,
                self.label_color,
                int(s * label.size),
                label.font,
                background_color=label.background,
                xpad=int(s * xpad),
                ypad=int(s * ypad),
                bold=label.bold,
                italic=label.italic,
                outline_width=int(s * label.outline_width),
            )
        else:
            from numpy import empty, uint8

            rgba = empty((1, 1, 4), uint8)
            rgba[0, 0, :] = self.label_color

        if rgba is None:
            label.session.logger.info("Can't find font for label")
        else:
            ih, iw = rgba.shape[:2]
            self._texture_size = (iw, ih)
            (x, y), (w, h) = self._placement
            from chimerax.graphics.drawing import rgba_drawing

            rgba_drawing(self, rgba, (x, y), (w, h), opaque=False)

    def _position_label_image(self):
        # Window has resized so update texture drawing placement
        (x, y), (w, h) = self._placement
        from chimerax.graphics.drawing import position_rgba_drawing

        position_rgba_drawing(self, (x, y), (w, h))

    @property
    def _placement(self):
        label = self.label
        # Convert 0-1 position to -1 to 1.
        x = -1 + (2 * label.xpos)
        y = -1 + (2 * label.ypos)
        y *= self._aspect
        w, h = [2 * s for s in self.size]  # Convert [0,1] size [-1,1] size.
        return (x, y), (w, h)

    @property
    def size(self):
        """Label size as fraction of window size (0-1)."""
        w, h = self._window_size
        label = self.label
        if not label.is_scalebar:
            tw, th = self._texture_size
            sx = tw / w
            sy = th / h
        else:
            sw, shp = label.scalebar_width, label.scalebar_height
            psize = self.view.pixel_size()
            if psize == 0:
                psize = 1  # No models open, so no center of rotation depth.
            tps = self._texture_pixel_scale
            sx, sy = tps * sw / (w * psize), tps * shp / h
        return sx, sy

    @property
    def label_color(self):
        label = self.label
        if label.color is None:
            if label.background is None:
                bg = [(255 * r) for r in label.view.background_color]
            else:
                bg = label.background
            from chimerax.core.colors import contrast_with

            if contrast_with([(c / 255) for c in bg[:3]])[0] == 0.0:
                rgba8 = (0, 0, 0, 255)
            else:
                rgba8 = (255, 255, 255, 255)
        else:
            rgba8 = tuple(label.color)
        return rgba8

    @property
    def overall_color(self):
        return self.label_color

    @overall_color.setter
    def overall_color(self, color):
        label = self.label
        label.color = color
        label.update_drawing()

    def x3d_needs(self, x3d_scene):
        from chimerax.core import x3d

        x3d_scene.need(x3d.Components.Text, 1)  # Text

    def custom_x3d(self, stream, x3d_scene, indent, place):
        # TODO
        pass

    def take_snapshot(self, session, flags):
        lattrs = (
            "name",
            "text",
            "color",
            "background",
            "size",
            "font",
            "bold",
            "italic",
            "xpos",
            "ypos",
            "visibility",
            "scalebar_width",
            "scalebar_height",
        )
        label = self.label
        lstate = {attr: getattr(label, attr) for attr in lattrs}
        data = {"label state": lstate, "version": 1}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        label = Label(session, **LabelModel._label_restore_parameters(data))
        return label.drawing

    @staticmethod
    def _label_restore_parameters(data):
        # Try to allow a newer session to open in older ChimeraX by
        # filtering out extra parameters not known by older ChimeraX.
        from inspect import signature

        param_names = signature(Label.__init__).parameters
        ls = data["label state"]
        params = {key: val for key, val in ls.items() if key in param_names}
        return params
