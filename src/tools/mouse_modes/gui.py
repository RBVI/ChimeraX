# vim: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    def __init__(self, session, tool_info):

        super().__init__(session, tool_info)

        self.mouse_modes = session.ui.main_window.graphics_window.mouse_modes
        self.button = 'right'

        self.icon_size = 48
        self.icon_border = 4
        self.rows = 1
        self.columns = 12

        panel_size = (300, self.rows * self.icon_size)
        from chimera.core.ui import MainToolWindow
        class MouseModesWindow(MainToolWindow):
            close_destroys = False
        tw = MouseModesWindow(self, size=panel_size)
        self.tool_window = tw
        parent = tw.ui_area

        from chimera.core import map, ui, markers
        from chimera.core.map import series
        import wx
        modes = (
            ui.SelectMouseMode,
            ui.RotateMouseMode,
            ui.TranslateMouseMode,
            ui.ZoomMouseMode,
            ui.TranslateSelectedMouseMode,
            ui.RotateSelectedMouseMode,
            map.ContourLevelMouseMode,
            map.PlanesMouseMode,
            markers.MarkerMouseMode,
            markers.MarkCenterMouseMode,
            markers.ConnectMouseMode,
            series.PlaySeriesMouseMode,
            )
        self.buttons = []
        for i, mode in enumerate(modes):
            location = ((i%self.columns)*self.icon_size,(i//self.columns)*self.icon_size)
            tb = wx.BitmapToggleButton(parent, i+1, self.bitmap(mode.icon_file), location)
            def button_press_cb(event, mode=mode, tb=tb):
                self.unset_other_buttons(tb)
                self.mouse_modes.bind_mouse_mode(self.button, mode(self.session))
            parent.Bind(wx.EVT_TOGGLEBUTTON, button_press_cb, id=i+1)
            tb.SetToolTip(wx.ToolTip(mode.name))
            self.buttons.append(tb)
        self.buttons[3].SetValue(True)          # Zoom id default mode

        tw.manage(placement="right", fixed_size = True)
        #tw.manage(placement="right")

        session.tools.add([self])

    def unset_other_buttons(self, button):
        for b in self.buttons:
            if b != button:
                b.SetValue(False)

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
