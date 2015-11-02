from chimera.core.errors import UserError as CommandError
def register_title_command():

    from chimera.core.commands import CmdDesc, register, BoolArg, IntArg, StringArg, FloatArg, ColorArg

    rargs = [('name', StringArg)]
    # Create and change have same arguments
    cargs = [('text', StringArg),
             ('color', ColorArg),
             ('size', IntArg),
             ('typeface', StringArg),
             ('xpos', FloatArg),
             ('ypos', FloatArg),
             ('visibility', BoolArg)]
    create_desc = CmdDesc(required = rargs, keyword = cargs)
    register('title create', create_desc, title_create)
    change_desc = CmdDesc(required = rargs, keyword = cargs)
    register('title change', change_desc, title_change)
    delete_desc = CmdDesc(required = rargs)
    register('title delete', delete_desc, title_delete)

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
        self.drawing = None
        self.make_drawing()

        lmap = getattr(session, 'labels', None)
        if lmap is None:
            session.labels = lmap = {}
        if name in lmap:
            lmap[name].delete()
        lmap[name] = self

    def make_drawing(self):
        v = self.session.main_view
        if self.color is None:
            light_bg = (sum(v.background_color[:3]) > 1.5)
            rgba8 = (0,0,0,255) if light_bg else (255,255,255,255)
        else:
            rgba8 = tuple(self.color.uint8x4())
        rgba = text_image_rgba(self.text, rgba8, self.size, self.typeface,
                               self.session.app_data_dir)
        if rgba is None:
            self.session.logger.info("Can't find font for title")
            return
        x,y = (-1 + 2*self.xpos, -1 + 2*self.ypos)    # Convert 0-1 position to -1 to 1.
        w,h = v.window_size
        uw,uh = 2*rgba.shape[1]/h, 2*rgba.shape[0]/h
        new = (self.drawing is None)
        from chimera.core.graphics.drawing import rgba_drawing
        self.drawing = d = rgba_drawing(rgba, (x, y), (uw, uh), self.drawing)
        d.display = self.visibility
        if new:
            v.add_overlay(d)
        return d

    def delete(self):
        d = self.drawing
        if d is None:
            return
        s = self.session
        s.main_view.remove_overlays([d])
        del s.labels[self.name]

def title_create(session, name, text = '', color = None, size = 24, typeface = 'Arial',
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
    return Label(**locals())

def title_change(session, name, text = None, color = None, size = None, typeface = None,
                 xpos = None, ypos = None, visibility = None):
    '''Change label parameters.'''
    l = session.labels[name]
    if not text is None: l.text = text
    if not color is None: l.color = color
    if not size is None: l.size = size
    if not typeface is None: l.typeface = typeface
    if not xpos is None: l.xpos = xpos
    if not ypos is None: l.ypos = ypos
    if not visibility is None: l.visibility = visibility
    l.make_drawing()
    return l

def title_delete(session, name):
    '''Delete label.'''
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
