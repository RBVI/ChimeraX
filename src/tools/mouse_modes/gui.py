# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    SIZE = (-1, 32)

    def __init__(self, session, tool_info):

        super().__init__(session, tool_info)

        self.mouse_modes = session.ui.main_window.graphics_window.mouse_modes

        tw = session.ui.create_main_tool_window(self, size=self.SIZE, destroy_hides=True)
        self.tool_window = tw
        parent = tw.ui_area

        import wx
        tb = wx.ToolBar(parent)
        mvmode = tb.AddTool(1, 'move models', self.bitmap('move.png'))
        parent.Bind(wx.EVT_TOOL, self.move_mode, mvmode)
        msmode = tb.AddTool(2, 'move selected models', self.bitmap('move_h2o.png'))
        parent.Bind(wx.EVT_TOOL, self.move_selected_mode, msmode)
        clmode = tb.AddTool(3, 'contour level', self.bitmap('contour.png'))
        parent.Bind(wx.EVT_TOOL, self.contour_mode, clmode)
        mpmode = tb.AddTool(4, 'crop map', self.bitmap('cubearrow.png'))
        parent.Bind(wx.EVT_TOOL, self.move_planes_mode, mpmode)
        sermode = tb.AddTool(5, 'play map series', self.bitmap('vseries.png'))
        parent.Bind(wx.EVT_TOOL, self.map_series_mode, sermode)
        tb.Realize()

        tw.manage(placement="right", fixed_size = True)

        session.tools.add([self])

    def move_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_move_mouse_mode(self.mouse_modes)

    def move_selected_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_move_selected_mouse_mode(self.mouse_modes)

    def contour_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_contour_mouse_mode(self.mouse_modes)

    def move_planes_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_move_planes_mouse_mode(self.mouse_modes)

    def map_series_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_map_series_mouse_mode(self.mouse_modes)

    def bitmap(self, filename, border = 3):
        width = height = self.SIZE[1] - 2*border
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
    # Override ToolInstance methods
    #
    def delete(self):
        s = self.session
        self.tool_window.shown = False
        self.tool_window.destroy()
        s.tools.remove([self])
        super().delete()

    def display(self, b):
        """Show or hide mouse mode panel."""
        self.tool_window.shown = b

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        pass

    def restore_snapshot(self, phase, session, version, data):
        pass

    def reset_state(self):
        pass
