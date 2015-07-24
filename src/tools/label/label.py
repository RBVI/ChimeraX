from chimera.core.cli import UserError as CommandError
def register_2dlabels_command():

    from chimera.core.cli import CmdDesc, register, BoolArg, IntArg, StringArg, FloatArg
    from chimera.core.color import ColorArg

    # Create and change have same arguments
    cargs = [('text', StringArg),
             ('color', ColorArg),
             ('size', IntArg),
             ('typeface', StringArg),
             ('xpos', FloatArg),
             ('ypos', FloatArg),
             ('visibility', BoolArg)]
    create_desc = CmdDesc(required = [('name', StringArg)], keyword = cargs)
    register('2dlabels create', create_desc, create_op)
    change_desc = CmdDesc(required = [('name', StringArg)], keyword = cargs)
    register('2dlabels change', change_desc, change_op)
    delete_desc = CmdDesc(required = [('name', StringArg)])
    register('2dlabels delete', delete_desc, delete_op)

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
        rgba = text_image_rgba(self.text, rgba8, self.size, self.typeface)
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

create_op = Label

def change_op(session, name, text = None, color = None, size = None, typeface = None,
              xpos = None, ypos = None, visibility = None):
    l = session.labels[name]
    if not text is None: l.text = text
    if not color is None: l.color = color
    if not size is None: l.size = size
    if not typeface is None: l.typeface = typeface
    if not xpos is None: l.xpos = xpos
    if not ypos is None: l.ypos = ypos
    if not visibility is None: l.visibility = visibility
    l.make_drawing()

def delete_op(session, name):
    l = session.labels[name]
    l.delete()

def text_image_rgba(text, color, size, typeface):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype('/Library/Fonts/%s.ttf' % typeface, size)
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
