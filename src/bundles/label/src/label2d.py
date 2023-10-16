# vim: set expandtab ts=4 sw=4:

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


# -----------------------------------------------------------------------------
#
def label2d(session, labels = None, *, text = None, color = None, bg_color = None,
            size = None, font = None, bold = None, italic = None,
            xpos = None, ypos = None, visibility = None, margin = None,
            outline = None, frames = None):
    keywords = ('text', 'color', 'bg_color', 'size', 'font', 'bold', 'italic',
                'xpos', 'ypos', 'visibility', 'margin', 'outline')
    loc = locals()
    kw = {attr:loc[attr] for attr in keywords if loc[attr] is not None}
    if labels is None:
        if text is None:
            labels = all_labels(session)
        else:
            return label_create(session, name = '', **kw)

    kw['frames'] = frames
    return [_update_label(session, l, **kw) for l in labels]

# -----------------------------------------------------------------------------
#
def label_create(session, name, text = '', color = None, bg_color = None,
                 size = 24, font = 'Arial', bold = None, italic = None,
                 xpos = 0.5, ypos = 0.5, visibility = True, margin = 0,
                 outline = 0):
    '''Create a label at a fixed position in the graphics window.

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
    '''
    if name == 'all':
        from chimerax.core.errors import UserError
        raise UserError("'all' is reserved to refer to all labels")
    elif name:
        lm = session_labels(session)
        if lm and lm.named_label(name) is not None:
            from chimerax.core.errors import UserError
            raise UserError('Label "%s" already exists' % name)

    kw = {'text':text, 'color':color, 'size':size, 'font':font,
          'bold':bold, 'italic':italic, 'xpos':xpos, 'ypos':ypos, 'visibility':visibility,
          'margin':margin, 'outline_width':outline}

    from chimerax.core.colors import Color
    if isinstance(color, Color):
        kw['color'] = color.uint8x4()
    elif isinstance(color, str) and color in ('default', 'auto'):
        kw['color'] = None

    if isinstance(bg_color, Color):
        kw['background'] = bg_color.uint8x4()
    elif isinstance(bg_color, str) and bg_color == 'none':
        kw['background'] = None

    has_graphics = session.main_view.render is not None
    if not has_graphics:
        from chimerax.core.errors import LimitationError
        raise LimitationError("Unable to draw 2D labels without rendering images")
        
    return Label(session, name, **kw)


# -----------------------------------------------------------------------------
#
def label_change(session, labels, *, text = None, color = None, bg_color = None,
                 size = None, font = None, bold = None, italic = None,
                 xpos = None, ypos = None, visibility = None, margin = None,
                 outline = None, frames = None):
    '''Change label parameters.'''
    kw = {'text':text, 'color':color, 'bg_color':bg_color, 'size':size, 'font':font,
          'bold':bold, 'italic':italic, 'xpos':xpos, 'ypos':ypos, 'visibility':visibility,
          'margin':margin, 'outline':outline, 'frames':frames}
    return [_update_label(session, l, **kw) for l in labels]


# -----------------------------------------------------------------------------
#
def _update_label(session, l, *, text = None, color = None, bg_color = None,
                 size = None, font = None, bold = None, italic = None,
                 xpos = None, ypos = None, visibility = None, margin = None,
                 outline = None, frames = None):
    if font is not None: l.font = font
    if bold is not None: l.bold = bold
    if text is not None: l.text = text
    if italic is not None: l.italic = italic
    if frames is None:
        if color is not None:
            l.color = None if color in ('default','auto') else color.uint8x4()
        if bg_color is not None:
            l.background = None if bg_color == 'none' else bg_color.uint8x4()
        if size is not None: l.size = size
        if xpos is not None: l.xpos = xpos
        if ypos is not None: l.ypos = ypos
        if visibility is not None: l.visibility = visibility
        if margin is not None: l.margin = margin
        if outline is not None: l.outline_width = outline
        l.update_drawing()
    else:
        _InterpolateLabel(session, l, color, bg_color, size, xpos, ypos, visibility, margin, outline, frames)


# -----------------------------------------------------------------------------
#
class _InterpolateLabel:
    def __init__(self, session, label, color, bg_color, size, xpos, ypos, visibility, margin, outline_width,
            frames):
        self.label = label
        from numpy import array_equal
        # even if color/background not changing, need color1/2 and bg1/2 for visibility changes
        from numpy import array, uint8
        self.orig_color1 = None if label.color is None else label.color.copy()
        self.color1, self.color2 = array(label.drawing.label_color, dtype=uint8), (color.uint8x4() if color else color)
        self.bg1, self.bg2 = (None if label.background is None else label.background.copy()), bg_color
        if color is None:
            # no change
            self.interp_color = False
        else:
            color2 = None if color == 'none' else color.uint8x4()
            if array_equal(label.color, color2):
                self.interp_color = False
            else:
                self.interp_color = True
        if bg_color is None:
            # no change
            self.interp_background = False
        else:
            bg2 = None if bg_color == 'none' else bg_color.uint8x4()
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
        fraction = frame / (self.frames-1)
        from numpy import array_equal
        if self.interp_color:
            if frame == self.frames-1:
                self.label.color = self.color2
            else:
                self.label.color = ((1 - fraction) * self.color1
                    + fraction * self.color2).astype(self.color1.dtype)
        if self.interp_background:
            if frame == self.frames-1:
                self.label.background = self.bg2
            else:
                self.label.background = ((1 - fraction) * self.bg1 + fraction * self.bg2).astype(self.bg1.dtype)
        if self.size2 is not None and self.size1 != self.size2:
            if frame == self.frames-1:
                self.label.size = self.size2
            else:
                self.label.size = (1 - fraction) * self.size1 + fraction * self.size2
        if self.xpos2 is not None and self.xpos1 != self.xpos2:
            if frame == self.frames-1:
                self.label.xpos = self.xpos2
            else:
                self.label.xpos = (1 - fraction) * self.xpos1 + fraction * self.xpos2
        if self.ypos2 is not None and self.ypos1 != self.ypos2:
            if frame == self.frames-1:
                self.label.ypos = self.ypos2
            else:
                self.label.ypos = (1 - fraction) * self.ypos1 + fraction * self.ypos2
        if self.visibility2 is not None and self.visibility1 != self.visibility2:
            if frame == self.frames-1:
                self.label.visibility = self.visibility2
                self.label.color = self.orig_color1 if self.color2 is None else self.color2
                self.label.background = self.bg1 if self.bg2 is None else self.bg2
            else:
                # fake gradual change in visibility via alpha channel
                if self.visibility2:
                    # becoming shown
                    self.label.color[-1] = self.label.color.dtype.type(self.color1[-1] * fraction)
                    if self.label.background is not None:
                        self.label.background[-1] = self.label.background.dtype.type(self.bg1[-1] * fraction)
                else:
                    # becoming hidden
                    self.label.color[-1] = self.label.color.dtype.type(self.color1[-1] * (1 - fraction))
                    if self.label.background is not None:
                        self.label.background[-1] = self.label.background.dtype.type(
                            self.bg1[-1] * (1 - fraction))
                self.label.visibility = True
        if self.margin2 is not None and self.margin1 != self.margin2:
            if frame == self.frames-1:
                self.label.margin = self.margin2
            else:
                self.label.margin = (1 - fraction) * self.margin1 + fraction * self.margin2
        if self.outline_width2 is not None and self.outline_width1 != self.outline_width2:
            if frame == self.frames-1:
                self.label.outline_width = self.outline_width2
            else:
                self.label.outline_width = (1 - fraction) * self.outline_width1 + fraction * self.outline_width2
        self.label.update_drawing()


# -----------------------------------------------------------------------------
#
def label_under_window_position(session, win_x, win_y):
    w,h = session.main_view.window_size
    fx,fy = (win_x+.5)/w, 1-(win_y+.5)/h    # win_y is 0 at top
    lm = session_labels(session)
    if lm is None:
        return None
    for lbl in lm.all_labels:
        dx,dy = fx - lbl.xpos, fy - lbl.ypos
        d = lbl.drawing
        if d.display and d.parents_displayed:
            lw,lh = d.size
            if dx >=0 and dx < lw and dy >=0 and dy < lh:
                return lbl
    return None
    
# -----------------------------------------------------------------------------
#
def label_delete(session, labels = None):
    '''Delete label.'''
    if labels is None or labels == 'all':
        lm = session_labels(session, create=False)
        labels = lm.all_labels if lm else []
        from .arrows import session_arrows
        am = session_arrows(session, create=False)
        labels = labels + (am.all_arrows if am else [])
    for l in tuple(labels):
        l.delete()

# -----------------------------------------------------------------------------
#
def label_listfonts(session):
    '''Report available fonts.'''
    has_graphics = session.main_view.render is not None
    if not has_graphics:
        from chimerax.core.errors import LimitationError
        raise LimitationError("Unable to do list fonts without being able to render images")
    from Qt.QtGui import QFontDatabase
    fnames = list(QFontDatabase.families())
    fnames.sort()
    session.logger.info('%d fonts available:\n%s' % (len(fnames), '\n'.join(fnames)))


# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class NamedLabelsArg(Annotation):

    name = "'all' or a 2d label name"
    _html_name = "<b>all</b> or a 2d label name"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            raise AnnotationError("Expected %s" % NamedLabelsArg.name)
        lm = session_labels(session)
        token, text, rest = next_token(text)
        if lm is None:
            raise AnnotationError("No label with name: '%s'" % token)
        if lm.named_label(token) is None:
            possible = [name for name in lm.label_names() if name.startswith(token)]
            if 'all'.startswith(token):
                possible.append('all')
            if not possible:
                raise AnnotationError("Unknown label identifier: '%s'" % token)
            possible.sort(key=len)
            token = possible[0]
        labels = lm.all_labels if token == 'all' else [lm.named_label(token)]
        return labels, token, rest


# -----------------------------------------------------------------------------
#
from chimerax.core.commands import ModelsArg
class LabelsArg(ModelsArg):
    """Parse command label model specifier"""
    name = "a label models specifier"

    @classmethod
    def parse(cls, text, session):
        models, text, rest = super().parse(text, session)
        labels = [m.label for m in models if isinstance(m, LabelModel)]
        return labels, text, rest

class LabelsArrowsArg(ModelsArg):
    """Parse command label/arrow model specifier"""
    name = "a label/arrow models specifier"

    @classmethod
    def parse(cls, text, session):
        from .arrows import ArrowModel
        models, text, rest = super().parse(text, session)
        labels = [m.label for m in models if isinstance(m, LabelModel)]
        arrows = [m.arrow for m in models if isinstance(m, ArrowModel)]
        return labels + arrows, text, rest

# -----------------------------------------------------------------------------
#
def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, Or, BoolArg, IntArg, StringArg, FloatArg, ColorArg
    from chimerax.core.commands import NonNegativeFloatArg, EnumOf, create_alias
    from .label3d import NoneArg

    labels_arg = [('labels', Or(NamedLabelsArg, LabelsArg))]
    # Create and change have same arguments
    cargs = [('text', StringArg),
             ('color', Or(EnumOf(['default', 'auto']), ColorArg)),
             ('bg_color', Or(NoneArg, ColorArg)),
             ('size', IntArg),
             ('font', StringArg),
             ('bold', BoolArg),
             ('italic', BoolArg),
             ('xpos', FloatArg),
             ('ypos', FloatArg),
             ('visibility', BoolArg),
             ('margin', NonNegativeFloatArg),
             ('outline', NonNegativeFloatArg)]
    create_desc = CmdDesc(required = [('name', StringArg)], keyword = cargs,
                          synopsis = 'Create a 2d label')
    register('2dlabels create', create_desc, label_create, logger=logger)
    change_desc = CmdDesc(required = labels_arg, keyword = cargs + [('frames', IntArg)],
                          synopsis = 'Change a 2d label')
    register('2dlabels change', change_desc, label_change, logger=logger)
    delete_desc = CmdDesc(optional = [('labels', Or(EnumOf(['all']), LabelsArrowsArg))],
                          synopsis = 'Delete a 2d label')
    register('2dlabels delete', delete_desc, label_delete, logger=logger)
    create_alias('~2dlabels', '2dlabels delete $*', logger=logger)
    fonts_desc = CmdDesc(synopsis = 'List available fonts')
    register('2dlabels listfonts', fonts_desc, label_listfonts, logger=logger)


    label_desc = CmdDesc(optional = labels_arg,
                         keyword = cargs + [('frames', IntArg)],
                          synopsis = 'Create or change a 2d label')
    register('2dlabels', label_desc, label2d, logger=logger)
    
# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class Labels(Model):
    def __init__(self, session):
        Model.__init__(self, '2D labels', session)
        self._labels = []	   
        self._named_labels = {}    # Map label name to Label object
        from chimerax.core.core_settings import settings
        self.handler = settings.triggers.add_handler('setting changed', self._background_color_changed)
        session.main_view.add_overlay(self)
        self.model_panel_show_expanded = False

    def delete(self):
        self.session.main_view.remove_overlays([self], delete = False)
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
        self._labels.remove(label)
        if len(self._labels) == 0:
            self.delete()

    def _background_color_changed(self, tname, tdata):
        # Update label color when graphics background color changes.
        if tdata[0] == 'background_color':
            for l in self.all_labels:
                if l.background is None:
                    l.update_drawing()
                    
    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        data = {'model state': Model.take_snapshot(self, session, flags),
                'version': 4}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        if 'model state' in data:
            s = Labels(session)
            Model.set_state_from_snapshot(s, session, data['model state'])
            session.models.add([s], root_model = True)
            if 'labels state' in data:
                # Older sessions restored the labels here.
                s.set_state_from_snapshot(session, data)
        else:
            # Restoring old session before 2d labels were Models.
            # Need to create labels model after all models created
            # so it does not use a model id already in use.
            def restore_old_labels(trigger_name, session, data=data):
                s = session_labels(session, create=True)
                s.set_state_from_snapshot(session, data)
                return 'delete handler'
            session.triggers.add_handler('end restore session', restore_old_labels)
            s = None
        return s

    def set_state_from_snapshot(self, session, data):
        self._labels = [Label(session, **ls) for ls in data['labels state']]


# -----------------------------------------------------------------------------
#
def find_label(session, name):
    lm = session_labels(session)
    return lm.named_label(name) if lm else None

# -----------------------------------------------------------------------------
#
def all_labels(session):
    lm = session_labels(session)
    return lm.all_labels if lm else []

# -----------------------------------------------------------------------------
#
def session_labels(session, create=False):
    lmlist = session.models.list(type = Labels)
    if lmlist:
        lm = lmlist[0]
    elif create:
        lm = Labels(session)
        session.models.add([lm], root_model = True)
    else:
        lm = None
    return lm

# -----------------------------------------------------------------------------
#
class Label:
    def __init__(self, session, name, text = '', color = None, background = None,
                 size = 24, font = 'Arial', bold = False, italic = False,
                 xpos = 0.5, ypos = 0.5, visibility = True, margin = 0, outline_width = 0,
                 scalebar_width = None, scalebar_height = None):

        self.session = session
        self.name = name
        self.text = text
        self.color = color
        from chimerax.core.colors import Color
        if isinstance(background, Color):
            self.background = background.uint8x4()
        else:
            # may already be numpy array if being restored from a session
            self.background = background
        self.size = size    	# Logical pixels.  On Mac retina display will render 2x more pixels.
        self.font = font
        self.bold = bold
        self.italic = italic
        self.xpos = xpos
        self.ypos = ypos
        self.visibility = visibility
        self.margin = margin	# Logical pixels.
        self.outline_width = outline_width
        self.scalebar_width = scalebar_width	# Angstroms
        self.scalebar_height = scalebar_height	# Pixels
        self.drawing = d = LabelModel(session, self)
        d.display = visibility
        lb = session_labels(session, create = True)
        lb.add_label(self)

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
        lm = session_labels(self.session)
        if lm:
            lm.delete_label(self)

    @property
    def is_scalebar(self):
        return self.scalebar_width is not None
    
# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class LabelModel(Model):

    pickable = False
    casts_shadows = False
    
    def __init__(self, session, label):
        name = label.name if label.name else label.text
        Model.__init__(self, name, session)
        self.label = label
        self._window_size = None	# Full window size in render pixels
        self._texture_size = None	# Label image size in render pixels
        self._texture_pixel_scale = 1	# Converts label.size from logical pixels to render pixels
        self._aspect = 1		# Scale y label positioning for image saving at non-screen aspect ratio
        self._last_placement = None	# Last location for scalebar, used for updating
        self.needs_update = True

    def delete(self):
        Model.delete(self)
        self.label.delete()
        
    def draw(self, renderer, draw_pass):
        if self._update_graphics(renderer):
            self.session.main_view.clear_drawing_changes()  # Avoid redrawing every frame.
        Model.draw(self, renderer, draw_pass)

    def _update_graphics(self, renderer):
        '''
        Recompute the label texture image or update its texture coordinates that
        position it in the window based on the rendered window size.
        When saving an image file the rendered size may differ from the on screen
        window size.  In that case make the label size match its relative size
        seen on screen.
        '''
        window_size = renderer.render_size()

        # Preserve on screen label size if saving an image of different size.
        if getattr(renderer, 'image_save', False):
            # When saving an image match the label's on screen fractional size
            # even though the image size in pixels may be different from on screen.
            sw,sh = self.session.main_view.window_size
            w,h = window_size
            pscale = (w / sw) if sw > 0 else 1
            aspect = (w*sh)/(h*sw) if h*sw > 0 else 1
        else:
            pscale = renderer.pixel_scale()
            aspect = 1
        if pscale != self._texture_pixel_scale or aspect != self._aspect:
            self._texture_pixel_scale = pscale
            self._aspect = aspect
            self.needs_update = True

        # Will need to reposition label if window size changes.
        win_size_changed = (window_size != self._window_size)
        if win_size_changed:
            self._window_size = window_size

        if self.needs_update:
            self.needs_update = False
            self._update_label_image()
        elif win_size_changed:
            self._position_label_image()
        elif self.label.is_scalebar and self._placement != self._last_placement:
            self._position_label_image()
            self._last_placement = self._placement
        else:
            return False

        return True

    def _update_label_image(self):
        l = self.label
        if not l.is_scalebar:
            xpad = (0 if l.background is None else int(.2 * l.size)) + l.margin
            ypad = l.margin
            s = self._texture_pixel_scale
            from chimerax.graphics import text_image_rgba
            rgba = text_image_rgba(l.text, self.label_color, int(s*l.size), l.font,
                                   background_color = l.background,
                                   xpad =int(s*xpad), ypad = int(s*ypad),
                                   bold = l.bold, italic = l.italic,
                                   outline_width=int(s*l.outline_width))
        else:
            from numpy import empty, uint8
            rgba = empty((1,1,4),uint8)
            rgba[0,0,:] = self.label_color
            
        if rgba is None:
            l.session.logger.info("Can't find font for label")
        else:
            ih, iw = rgba.shape[:2]
            self._texture_size = (iw,ih)
            (x,y),(w,h) = self._placement
            from chimerax.graphics.drawing import rgba_drawing
            rgba_drawing(self, rgba, (x,y), (w,h), opaque = False)

    def _position_label_image(self):
        # Window has resized so update texture drawing placement
        (x,y),(w,h) = self._placement
        from chimerax.graphics.drawing import position_rgba_drawing
        position_rgba_drawing(self, (x,y), (w,h))

    @property
    def _placement(self):
        l = self.label
        x,y = (-1 + 2*l.xpos, -1 + 2*l.ypos)    # Convert 0-1 position to -1 to 1.
        y *= self._aspect
        w,h = [2*s for s in self.size]		# Convert [0,1] size [-1,1] size.
        return (x,y), (w,h)

    @property
    def size(self):
        '''Label size as fraction of window size (0-1).'''
        w,h = self._window_size
        l = self.label
        if not l.is_scalebar:
            tw,th = self._texture_size
            sx,sy = (tw/w, th/h)
        else:
            sw,shp = l.scalebar_width, l.scalebar_height
            psize = self.session.main_view.pixel_size()
            if psize == 0:
                psize = 1	# No models open, so no center of rotation depth.
            tps = self._texture_pixel_scale
            sx,sy = tps*sw/(w*psize), tps*shp/h
        return sx,sy
    
    @property
    def label_color(self):
        l = self.label
        if l.color is None:
            if l.background is None:
                bg = [255*r for r in l.session.main_view.background_color]
            else:
                bg = l.background
            from chimerax.core.colors import contrast_with
            if contrast_with([c/255 for c in bg[:3]])[0] == 0.0:
                rgba8 = (0, 0, 0, 255)
            else:
                rgba8 = (255, 255, 255, 255)
        else:
            rgba8 = tuple(l.color)
        return rgba8

    def _get_model_color(self):
        return self.label_color
    def _set_model_color(self, color):
        l = self.label
        l.color = color
        l.update_drawing()
    model_color = property(_get_model_color, _set_model_color)

    def x3d_needs(self, x3d_scene):
        from .. import x3d
        x3d_scene.need(x3d.Components.Text, 1)  # Text

    def custom_x3d(self, stream, x3d_scene, indent, place):
        # TODO
        pass

    def take_snapshot(self, session, flags):
        lattrs = ('name', 'text', 'color', 'background', 'size', 'font',
                  'bold', 'italic', 'xpos', 'ypos', 'visibility',
                  'scalebar_width', 'scalebar_height')
        l = self.label
        lstate = {attr:getattr(l, attr) for attr in lattrs}
        data = {'label state': lstate,
                'version': 1}
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
        ls = data['label state']
        params = {key:val for key,val in ls.items() if key in param_names}
        return params 
        
