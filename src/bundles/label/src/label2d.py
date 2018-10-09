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


# -----------------------------------------------------------------------------
#
def label_create(session, name, text = '', color = None,
                 size = 24, font = 'Arial', bold = False, italic = False,
                 xpos = 0.5, ypos = 0.5, visibility = True):
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
    '''
    if name == 'all':
        from chimerax.core.errors import UserError
        raise UserError("'all' is reserved to refer to all labels")
    kw = locals().copy()
    if color is not None:
        kw['color'] = color.uint8x4()
    return Label(**kw)


# -----------------------------------------------------------------------------
#
def label_change(session, name, text = None, color = None,
                 size = None, font = None, bold = False, italic = False,
                 xpos = None, ypos = None, visibility = None):
    '''Change label parameters.'''
    lb = session_labels(session)
    if name == 'all':
        for n in lb.labels.keys():
            label_change(session, n, text, color, size, font, xpos, ypos, visibility)
        return
    l = lb.labels[name]
    if not text is None: l.text = text
    if not color is None: l.color = color.uint8x4()
    if not size is None: l.size = size
    if not font is None: l.font = font
    if not bold is None: l.bold = bold
    if not italic is None: l.italic = italic
    if not xpos is None: l.xpos = xpos
    if not ypos is None: l.ypos = ypos
    if not visibility is None: l.visibility = visibility
    l.update_drawing()
    return l

# -----------------------------------------------------------------------------
#
def label_under_window_position(session, win_x, win_y):
    w,h = session.main_view.window_size
    fx,fy = (win_x+.5)/w, 1-(win_y+.5)/h	# win_y is 0 at top
    for lbl in session_labels(session).labels.values():
        dx,dy = fx - lbl.xpos, fy - lbl.ypos
        lw,lh = lbl.drawing.size
        if dx >=0 and dx < lw and dy >=0 and dy < lh:
            return lbl
    return None
    
# -----------------------------------------------------------------------------
#
def label_delete(session, name):
    '''Delete label.'''
    lb = session_labels(session)
    if name == 'all':
        for l in tuple(lb.labels.values()):
            l.delete()
        return
    l = lb.labels[name]
    l.delete()

# -----------------------------------------------------------------------------
#
def label_listfonts(session):
    '''Report available fonts.'''
    from PyQt5.QtGui import QFontDatabase
    fdb = QFontDatabase()
    fnames = list(fdb.families())
    fnames.sort()
    session.logger.info('%d fonts available:\n%s' % (len(fnames), '\n'.join(fnames)))


# -----------------------------------------------------------------------------
#
def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, IntArg, StringArg, FloatArg, ColorArg

    rargs = [('name', StringArg)]
    existing_arg = [('name', NameArg)]
    # Create and change have same arguments
    cargs = [('text', StringArg),
             ('color', ColorArg),
             ('size', IntArg),
             ('font', StringArg),
             ('bold', BoolArg),
             ('italic', BoolArg),
             ('xpos', FloatArg),
             ('ypos', FloatArg),
             ('visibility', BoolArg)]
    create_desc = CmdDesc(required = rargs, keyword = cargs,
                          synopsis = 'Create a 2d label')
    register('2dlabels create', create_desc, label_create, logger=logger)
    change_desc = CmdDesc(required = existing_arg, keyword = cargs,
                          synopsis = 'Change an existing 2d label')
    register('2dlabels change', change_desc, label_change, logger=logger)
    delete_desc = CmdDesc(required = existing_arg,
                          synopsis = 'Delete a 2d label')
    register('2dlabels delete', delete_desc, label_delete, logger=logger)
    fonts_desc = CmdDesc(synopsis = 'List available fonts')
    register('2dlabels listfonts', fonts_desc, label_listfonts, logger=logger)


# -----------------------------------------------------------------------------
#
from chimerax.core.state import StateManager
class Labels(StateManager):
    def __init__(self):
        StateManager.__init__(self)
        self.labels = {}	# Map label name to Label object

    def add(self, label):
        n = label.name
        ls = self.labels
        if n in ls:
            ls[n].delete()
        ls[n] = label

    def delete(self, label):
        del self.labels[label.name]

    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        lattrs = ('name', 'text', 'color', 'size', 'font',
                  'bold', 'italic', 'xpos', 'ypos', 'visibility')
        lstate = tuple({attr:getattr(l, attr) for attr in lattrs}
                       for l in self.labels.values())
        data = {'labels state': lstate, 'version': 1}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = session_labels(session)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        self.labels = {ls['name']:Label(session, **ls) for ls in data['labels state']}

    def reset_state(self, session):
        for l in tuple(self.labels.values()):
            l.delete()
        self.labels = {}


# -----------------------------------------------------------------------------
#
def session_labels(session):
    lb = getattr(session, '_2d_labels', None)
    if lb is None:
        session._2d_labels = lb = Labels()
    return lb
        

# -----------------------------------------------------------------------------
#
class Label:
    def __init__(self, session, name, text = '', color = None,
                 size = 24, font = 'Arial', bold = False, italic = False,
                 xpos = 0.5, ypos = 0.5, visibility = True):
        self.session = session
        self.name = name
        self.text = text
        self.color = color
        self.size = size	# Points (1/72 inch) to get similar appearance on high DPI displays
        self.font = font
        self.bold = bold
        self.italic = italic
        self.xpos = xpos
        self.ypos = ypos
        self.visibility = visibility
        self.drawing = d = LabelDrawing(self)
        session.main_view.add_overlay(d)

        lb = session_labels(session)
        lb.add(self)

    def update_drawing(self):
        d = self.drawing
        d.needs_update = True
        d.redraw_needed()
        
    def delete(self):
        d = self.drawing
        if d is None:
            return
        s = self.session
        s.main_view.remove_overlays([d])
        session_labels(s).delete(self)


# -----------------------------------------------------------------------------
#
from chimerax.core.graphics.drawing import Drawing
class LabelDrawing(Drawing):

    pickable = False
    casts_shadows = False
    
    def __init__(self, label):
        Drawing.__init__(self, 'label %s' % label.name)
        self.label = label
        self.window_size = None
        self.texture_size = None
        self.needs_update = True
        
    def draw(self, renderer, draw_pass):
        if not self.update_drawing():
            self.resize()
        Drawing.draw(self, renderer, draw_pass)

    def update_drawing(self):
        if not self.needs_update:
            return False
        self.needs_update = False
        l = self.label
        v = l.session.main_view
        if l.color is None:
            light_bg = (sum(v.background_color[:3]) > 1.5)
            rgba8 = (0,0,0,255) if light_bg else (255,255,255,255)
        else:
            rgba8 = tuple(l.color)
        from chimerax.core.graphics import text_image_rgba
        rgba = text_image_rgba(l.text, rgba8, l.size, l.font,
                               bold = l.bold, italic = l.italic)
        if rgba is None:
            l.session.logger.info("Can't find font for label")
            return True
        self.set_text_image(rgba)
        self.display = l.visibility
        return True
        
    def set_text_image(self, rgba):
        l = self.label
        x,y = (-1 + 2*l.xpos, -1 + 2*l.ypos)    # Convert 0-1 position to -1 to 1.
        v = l.session.main_view
        self.window_size = w,h = v.window_size
        th, tw = rgba.shape[:2]
        self.texture_size = (tw,th)
        uw,uh = 2*tw/w, 2*th/h
        from chimerax.core.graphics.drawing import rgba_drawing
        rgba_drawing(self, rgba, (x, y), (uw, uh), opaque = False)

    @property
    def size(self):
        '''Label size as fraction of window size (0-1).'''
        w,h = self.window_size
        tw,th = self.texture_size
        return (tw/w, th/h)

    def resize(self):
        l = self.label
        v = l.session.main_view
        if v.window_size != self.window_size:
            # Window has resized so update texture drawing size
            self.window_size = w,h = v.window_size
            tw,th = self.texture_size
            uw,uh = 2*tw/w, 2*th/h
            x,y = (-1 + 2*l.xpos, -1 + 2*l.ypos)    # Convert 0-1 position to -1 to 1.
            from chimerax.core.graphics.drawing import position_rgba_drawing
            position_rgba_drawing(self, (x,y), (uw,uh))

    def x3d_needs(self, x3d_scene):
        from .. import x3d
        x3d_scene.need(x3d.Components.Text, 1)  # Text

    def custom_x3d(self, stream, x3d_scene, indent, place):
        # TODO
        pass

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class NameArg(Annotation):

    name = "'all' or a label identifier"
    _html_name = "<b>all</b> or a label identifier"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            raise AnnotationError("Expected %s" % NameArg.name)
        lmap = session_labels(session).labels
        token, text, rest = next_token(text)
        if token not in lmap:
            possible = [name for name in lmap if name.startswith(token)]
            if 'all'.startswith(token):
                possible.append('all')
            if not possible:
                raise AnnotationError("Unknown label identifier: '%s'" % token)
            possible.sort(key=len)
            token = possible[0]
        return token, token, rest

