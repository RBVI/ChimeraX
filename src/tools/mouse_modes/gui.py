# vim: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    def __init__(self, session, tool_info):

        super().__init__(session, tool_info)

        self.mouse_modes = session.ui.main_window.graphics_window.mouse_modes
        self.button_to_bind = 'right'

        self.icon_size = 48
        self.max_icon_size = 48
        self.min_icon_size = 24
        self.icon_border = 4

        min_panel_width = self.icon_size
        panel_height = self.icon_size
        panel_size = (min_panel_width, panel_height)

        from chimera.core.ui import MainToolWindow
        class MouseModesWindow(MainToolWindow):
            close_destroys = False

        self.tool_window = tw = MouseModesWindow(self, size=panel_size)
        parent = tw.ui_area

        from chimera.core import map, ui, markers
        from chimera.core.map import series
        self.modes = (
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
        initial_mode = ui.ZoomMouseMode

        self.buttons = self.create_buttons(self.modes, self.button_to_bind,
                                           initial_mode, parent, session)

        tw.manage(placement="right", fixed_size = True)

        import wx
        parent.Bind(wx.EVT_SIZE, self.resize_cb)

        session.tools.add([self])

    def create_buttons(self, modes, button_to_bind, initial_mode, parent, session):
        import wx
        buttons = []
        for i, mode in enumerate(modes):
            tb = wx.BitmapToggleButton(parent, i+1, self.bitmap(mode.icon_file))
            def button_press_cb(event, mode=mode, tb=tb):
                self.unset_other_buttons(tb)
                self.mouse_modes.bind_mouse_mode(button_to_bind, mode(session))
            parent.Bind(wx.EVT_TOGGLEBUTTON, button_press_cb, id=i+1)
            tb.SetToolTip(wx.ToolTip(mode.name))
            buttons.append(tb)
            if mode == initial_mode:
                tb.SetValue(True)
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
        for i,b in enumerate(self.buttons):
            b.SetBitmap(self.bitmap(self.modes[i].icon_file))
            b.SetSize((icon_size,icon_size))
            pos = ((i%columns)*icon_size,(i//columns)*icon_size)
            b.SetPosition(pos)
        self.icon_size = icon_size

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
    def take_snapshot(self, session, flags):
        pass

    def restore_snapshot(self, phase, session, version, data):
        pass

    def reset_state(self, session):
        pass
