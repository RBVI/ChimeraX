# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    SESSION_ENDURING = True

    def __init__(self, session, bundle_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, bundle_info)

        self.mouse_modes = session.ui.main_window.graphics_window.mouse_modes
        self.button_to_bind = 'right'

        self.icon_size = 48
        self.max_icon_size = 48
        self.min_icon_size = 24
        self.icon_border = 4

        min_panel_width = self.icon_size
        panel_height = self.icon_size
        panel_size = (min_panel_width, panel_height)

        from chimerax.core.ui import MainToolWindow
        class MouseModesWindow(MainToolWindow):
            close_destroys = False

        self.tool_window = tw = MouseModesWindow(self, size=panel_size)
        parent = tw.ui_area

        mm = session.ui.main_window.graphics_window.mouse_modes
        self.modes = [m for m in mm.modes if m.icon_file]
        initial_mode = [m for m in self.modes if m.name == 'zoom'][0]

        self.buttons = self.create_buttons(self.modes, self.button_to_bind,
                                           initial_mode, parent, session)

        tw.manage(placement="right", fixed_size = True)

        import wx
        parent.Bind(wx.EVT_SIZE, self.resize_cb)

    def create_buttons(self, modes, button_to_bind, initial_mode, parent, session):
        import wx
        buttons = []
        for i, mode in enumerate(modes):
            tb = wx.BitmapToggleButton(parent, i+1, self.bitmap(mode.icon_file))
            def button_press_cb(event, mode=mode, tb=tb):
                self.unset_other_buttons(tb)
                modifiers = []
                self.mouse_modes.bind_mouse_mode(button_to_bind, modifiers, mode)
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
        data = {"shown": self.tool_window.shown}
        return self.bundle_info.session_write_version, data

    @classmethod
    def restore_snapshot_new(cls, session, bundle_info, version, data):
        return cls.get_singleton(session)

    def restore_snapshot_init(self, session, bundle_info, version, data):
        if version not in bundle_info.session_versions:
            from chimerax.core.state import RestoreError
            raise RestoreError("unexpected version")
        self.display(data["shown"])

    def reset_state(self, session):
        pass

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, MouseModePanel, 'mouse_modes')
