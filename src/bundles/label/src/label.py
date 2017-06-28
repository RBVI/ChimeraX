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

from chimerax.core.errors import UserError as CommandError
from chimerax.core.commands import Annotation, AnnotationError, next_token

class NameArg(Annotation):

    name = "'all' or a label identifier"
    _html_name = "<b>all</b> or a label identifier"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % NameArg.name)
        lmap = getattr(session, 'labels', {})
        if len(lmap) == 0:
            raise AnnotationError("No labels exist")
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


def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, IntArg, StringArg, FloatArg, ColorArg

    rargs = [('name', StringArg)]
    existing_arg = [('name', NameArg)]
    # Create and change have same arguments
    cargs = [('text', StringArg),
             ('color', ColorArg),
             ('size', IntArg),
             ('typeface', StringArg),
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

class Label:
    def __init__(self, session, name, text = '', color = None, size = 24, typeface = 'Arial',
                 xpos = 0.5, ypos = 0.5, visibility = True):
        self.session = session
        self.name = name
        self.text = text
        self.color = color
        self.size = size
        self.typeface = typeface
        self.xpos = xpos
        self.ypos = ypos
        self.visibility = visibility
        self.drawing = d = LabelDrawing(self)
        session.main_view.add_overlay(d)

        lmap = getattr(session, 'labels', None)
        if lmap is None:
            session.labels = lmap = {}
        if name in lmap:
            lmap[name].delete()
        lmap[name] = self

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
        del s.labels[self.name]

from chimerax.core.graphics.drawing import Drawing
class LabelDrawing(Drawing):

    pickable = False
    
    def __init__(self, label):
        Drawing.__init__(self, 'label %s' % label.name)
        self.label = label
        self.window_size = None
        self.texture_size = None
        self.needs_update = True
        
    def draw(self, renderer, place, draw_pass, selected_only=False):
        if not self.update_drawing():
            self.resize()
        Drawing.draw(self, renderer, place, draw_pass, selected_only)

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
            rgba8 = tuple(l.color.uint8x4())
        from chimerax import app_data_dir
        rgba = text_image_rgba(l.text, rgba8, l.size, l.typeface, app_data_dir)
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
        rgba_drawing(self, rgba, (x, y), (uw, uh))

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


def label_create(session, name, text = '', color = None, size = 24, typeface = 'Arial',
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
      Font size in pixels.
    typeface : string
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
        raise CommandError("'all' is reserved to refer to all labels")
    return Label(**locals())

def label_change(session, name, text = None, color = None, size = None, typeface = None,
                 xpos = None, ypos = None, visibility = None):
    '''Change label parameters.'''
    if name == 'all':
        for n in session.labels.keys():
            label_change(session, n, text, color, size, typeface, xpos, ypos, visibility)
        return
    l = session.labels[name]
    if not text is None: l.text = text
    if not color is None: l.color = color
    if not size is None: l.size = size
    if not typeface is None: l.typeface = typeface
    if not xpos is None: l.xpos = xpos
    if not ypos is None: l.ypos = ypos
    if not visibility is None: l.visibility = visibility
    l.update_drawing()
    return l

def label_delete(session, name):
    '''Delete label.'''
    if name == 'all':
        for l in tuple(session.labels.values()):
            l.delete()
        return
    l = session.labels[name]
    l.delete()

def text_image_rgba(text, color, size, typeface, data_dir):
    import os, sys
    from PIL import Image, ImageDraw, ImageFont
    font_dir = os.path.join(data_dir, 'fonts', 'freefont')
    f = None
    for tf in (typeface, 'FreeSans'):
        path = os.path.join(font_dir, '%s.ttf' % tf)
        if os.path.exists(path):
            f = ImageFont.truetype(path, size)
            break
        if sys.platform.startswith('darwin'):
            path = '/Library/Fonts/%s.ttf' % tf
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                break
    if f is None:
        return
    pixel_size = f.getsize(text)
    # Size 0 image gives rgba array that is not 3-dimensional
    pixel_size = (max(1,pixel_size[0]), max(1,pixel_size[1]))
    i = Image.new('RGBA', pixel_size)
    d = ImageDraw.Draw(i)
    #print('Size of "%s" is %s' % (text, pixel_size))
    d.text((0,0), text, font = f, fill = color)
    #i.save('test.png')
    from numpy import array
    rgba = array(i)
#    print ('Text "%s" rgba array size %s' % (text, tuple(rgba.shape)))
    frgba = rgba[::-1,:,:]	# Flip so text is right side up.
    return frgba
