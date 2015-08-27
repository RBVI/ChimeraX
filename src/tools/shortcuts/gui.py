# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ShortcutPanel(ToolInstance):

    def __init__(self, session, tool_info):

        super().__init__(session, tool_info)

        from .shortcuts import keyboard_shortcuts
        self.keyboard_shortcuts = keyboard_shortcuts(session)

        self.icon_size = 48
        self.icon_border = 4
        self.rows = 1
        self.columns = 6

        panel_size = (300, self.rows * self.icon_size)
        from chimera.core.ui import MainToolWindow
        class ShortcutWindow(MainToolWindow):
            close_destroys = False
        tw = ShortcutWindow(self, size=panel_size)
        self.tool_window = tw
        parent = tw.ui_area

        from chimera.core import map, ui, markers
        from chimera.core.map import series
        import wx
        shortcuts = (
            ('st', 'stick.png', 'Show molecule in stick style'),
            ('sp', 'sphere.png', 'Show molecule in sphere style'),
            ('bs', 'ball.png', 'Show molecule in ball and stick style'),
            ('rb', 'ribbon.png', 'Show molecule in ribbon style'),
            ('ms', 'surf.png', 'Show molecular surface'),
            )
        self.buttons = []
        for i, (keys, icon_file, descrip) in enumerate(shortcuts):
            location = ((i%self.columns)*self.icon_size,(i//self.columns)*self.icon_size)
            tb = wx.BitmapButton(parent, i+1, self.bitmap(icon_file), location)
            def button_press_cb(event, keys=keys, ks=self.keyboard_shortcuts):
                ks.run_shortcut(keys)
            parent.Bind(wx.EVT_BUTTON, button_press_cb, id=i+1)
            tb.SetToolTip(wx.ToolTip(descrip))
            self.buttons.append(tb)

        tw.manage(placement="right", fixed_size = True)

        session.tools.add([self])

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

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        pass

    def restore_snapshot(self, phase, session, version, data):
        pass

    def reset_state(self):
        pass

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    running = session.tools.find_by_class(ShortcutPanel)
    if len(running) > 1:
        raise RuntimeError("Can only have one shortcut panel")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('shortcuts')
            return ShortcutPanel(session, tool_info)
        else:
            return None
    else:
        return running[0]
