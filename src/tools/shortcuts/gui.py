# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ShortcutPanel(ToolInstance):

    shortcuts = []
    SESSION_ENDURING = True

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        from .shortcuts import keyboard_shortcuts
        self.keyboard_shortcuts = keyboard_shortcuts(session)

        self.icon_size = 48
        self.max_icon_size = 48
        self.min_icon_size = 24
        self.icon_border = 4

        columns = 12
        rows = (len(self.shortcuts) + columns - 1)//columns
        min_panel_width = self.icon_size
        panel_height = rows * self.icon_size
        panel_size = (min_panel_width, panel_height)

        from chimerax.core.ui.gui import MainToolWindow
        class ShortcutWindow(MainToolWindow):
            close_destroys = False

        self.tool_window = tw = ShortcutWindow(self, size=panel_size)
        parent = tw.ui_area

        self.buttons = self.create_buttons(parent)

        import wx
        parent.Bind(wx.EVT_SIZE, self.resize_cb)

        tw.manage(placement="right", fixed_size = True)

    def create_buttons(self, parent):

        import wx
        buttons = []
        for i, (keys, icon_file, descrip) in enumerate(self.shortcuts):
            tb = wx.BitmapButton(parent, i+1, self.bitmap(icon_file))
            tb.icon_file = icon_file
            def button_press_cb(event, keys=keys, ks=self.keyboard_shortcuts):
                ks.run_shortcut(keys)
            parent.Bind(wx.EVT_BUTTON, button_press_cb, id=i+1)
            tb.SetToolTip(wx.ToolTip(descrip))
            buttons.append(tb)
        return buttons

    def resize_cb(self, event):
        size = event.GetSize()
        w, h = size.GetWidth(), size.GetHeight()
        icon_size = min(self.max_icon_size, max(self.min_icon_size, w // len(self.buttons)))
        if icon_size == self.icon_size:
            return

        n = len(self.buttons)
        num_per_row = w//icon_size
        rows = max(1, h//icon_size)
        columns = (n + rows - 1) // rows
        self.resize_buttons(columns, icon_size)

        # TODO: Try resizing pane height
        # self.tool_window.ui_area.SetSize((w,100))

    def resize_buttons(self, columns, icon_size):
        self.icon_size = icon_size
        for i,b in enumerate(self.buttons):
            b.SetBitmap(self.bitmap(b.icon_file))
            b.SetSize((icon_size,icon_size))
            pos = ((i%columns)*icon_size,(i//columns)*icon_size)
            b.SetPosition(pos)

    def bitmap(self, filename):
        width = height = self.icon_size - 2*self.icon_border
        from os import path
        icondir = path.join(path.dirname(__file__), 'icons')
        import wx
        bitmap = wx.Bitmap(path.join(icondir, filename))
        image = bitmap.ConvertToImage()
        image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        result = wx.Bitmap(image)
        return result

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, cls.tool_name)

class MoleculeDisplayPanel(ShortcutPanel):
    tool_name = 'molecule_display_shortcuts'
    shortcuts = (
        ('da', 'atomshow.png', 'Show atoms'),
        ('ha', 'atomhide.png', 'Hide atoms'),
        ('rb', 'ribshow.png', 'Show molecule ribbons'),
        ('hr', 'ribhide.png', 'Hide molecule ribbons'),
        ('ms', 'surfshow.png', 'Show molecular surface'),
        ('hs', 'surfhide.png', 'Hide molecular surface'),
        ('st', 'stick.png', 'Show molecule in stick style'),
        ('sp', 'sphere.png', 'Show molecule in sphere style'),
        ('bs', 'ball.png', 'Show molecule in ball and stick style'),
        ('ce', 'colorbyelement.png', 'Color atoms by element'),
        ('cc', 'colorbychain.png', 'Color atoms by chain'),
        ('rc', 'colorrandom.png', 'Random atom colors'),
    )
    help = "help:user/tools/moldisplay.html"

class GraphicsPanel(ShortcutPanel):
    tool_name = 'graphics_shortcuts'
    shortcuts = (
        ('wb', 'whitebg.png', 'White background'),
        ('gb', 'graybg.png', 'Gray background'),
        ('bk', 'blackbg.png', 'Black background'),
        ('ls', 'simplelight.png', 'Simple lighting'),
        ('la', 'softlight.png', 'Soft lighting'),
        ('lf', 'fulllight.png', 'Full lighting'),
        ('lF', 'flat.png', 'Flat lighting'),
        ('se', 'silhouette.png', 'Silhouette edges'),
        ('va', 'viewall.png', 'View all'),
        ('dv', 'orient.png', 'Standard orientation'),
        ('sx', 'camera.png', 'Save snapshot to desktop'),
        ('vd', 'video.png', 'Record spin movie'),
    )
    help = "help:user/tools/graphics.html"

panel_classes = {
    'molecule_display_shortcuts': MoleculeDisplayPanel,
    'graphics_shortcuts': GraphicsPanel,
}
