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
        buttons = (('zoom', 'zoom.png', self.zoom_mode),
                   ('move selected models', 'move_h2o.png', self.move_selected_mode),
                   ('rotate selected models', 'rotate_h2o.png', self.rotate_selected_mode),
                   ('contour level', 'contour.png', self.contour_mode),
                   ('move planes', 'cubearrow.png', self.move_planes_mode),
                   ('play map series', 'vseries.png', self.map_series_mode),
               )
        self.buttons = []
        size = self.SIZE[1]
        for i, (name, icon, callback) in enumerate(buttons):
            tb = wx.BitmapToggleButton(parent, i+1, self.bitmap(icon), (i*size,0))
            def button_press_cb(event, cb=callback, tb=tb):
                self.unset_other_buttons(tb)
                cb(event)
            parent.Bind(wx.EVT_TOGGLEBUTTON, button_press_cb, id=i+1)
            tb.SetToolTip(wx.ToolTip(name))
            self.buttons.append(tb)
        self.buttons[0].SetValue(True)

        tw.manage(placement="right", fixed_size = True)

        session.tools.add([self])

    def unset_other_buttons(self, button):
        for b in self.buttons:
            if b != button:
                b.SetValue(False)

    def zoom_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_zoom_mouse_mode(self.mouse_modes)

    def move_selected_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_translate_selected_mouse_mode(self.mouse_modes)

    def rotate_selected_mode(self, event):
        from chimera.core import shortcuts
        shortcuts.enable_rotate_selected_mouse_mode(self.mouse_modes)

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
    def take_snapshot(self, phase, session, flags):
        pass

    def restore_snapshot(self, phase, session, version, data):
        pass

    def reset_state(self):
        pass
