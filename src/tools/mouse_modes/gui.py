# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    SESSION_ENDURING = True
    help = "help:user/tools/mousemodes.html"

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.mouse_modes = session.ui.main_window.graphics_window.mouse_modes
        self.button_to_bind = 'right'

        from chimerax.core import window_sys
        if window_sys == "wx":
            self.icon_size = 48
            self.max_icon_size = 48
            self.min_icon_size = 24
            self.icon_border = 4

            min_panel_width = self.icon_size
            panel_height = self.icon_size
            panel_size = (min_panel_width, panel_height)

            from chimerax.core.ui.gui import MainToolWindow
            class MouseModesWindow(MainToolWindow):
                close_destroys = False

            self.tool_window = tw = MouseModesWindow(self, size=panel_size)
            parent = tw.ui_area

            import wx
            parent.Bind(wx.EVT_SIZE, self.resize_cb)
        else:
            parent = session.ui.main_window

        mm = session.ui.main_window.graphics_window.mouse_modes
        self.modes = [m for m in mm.modes if m.icon_file]
        initial_mode = [m for m in self.modes if m.name == 'zoom'][0]

        self.buttons = self.create_buttons(self.modes, self.button_to_bind,
                                           initial_mode, parent, session)

        if window_sys == "wx":
            tw.manage(placement="right", fixed_size = True)

    def create_buttons(self, modes, button_to_bind, initial_mode, parent, session):
        from chimerax.core import window_sys
        if window_sys == "wx":
            import wx
            buttons = []
            for i, mode in enumerate(modes):
                tb = wx.BitmapToggleButton(parent, i+1, self.bitmap(mode.icon_file))
                def button_press_cb(event, mode=mode, tb=tb):
                    self.unset_other_buttons(tb)
                    mname = mode.name
                    if ' ' in mname:
                        mname = '"%s"' % mname
                    from chimerax.core.commands import run
                    run(self.session, 'mousemode %s %s' % (button_to_bind, mname))
                parent.Bind(wx.EVT_TOGGLEBUTTON, button_press_cb, id=i+1)
                tb.SetToolTip(wx.ToolTip(mode.name))
                buttons.append(tb)
                if mode == initial_mode:
                    tb.SetValue(True)
            return buttons
        else:
            from PyQt5.QtWidgets import QAction, QToolBar, QActionGroup
            from PyQt5.QtGui import QIcon
            from PyQt5.QtCore import Qt, QSize
            tb = QToolBar(self.display_name, parent)
            tb.setIconSize(QSize(40,40))
            parent.addToolBar(Qt.RightToolBarArea, tb)
            group = QActionGroup(tb)
            for mode in modes:
                from os import path
                icon_dir = path.join(path.dirname(__file__), 'icons')
                action = QAction(QIcon(path.join(icon_dir, mode.icon_file)), mode.name, group)
                action.setCheckable(True)
                def button_press_cb(event, mode=mode):
                    mname = mode.name
                    if ' ' in mname:
                        mname = '"%s"' % mname
                    from chimerax.core.commands import run
                    run(self.session, 'mousemode %s %s' % (button_to_bind, mname))
                action.triggered.connect(button_press_cb)
                group.addAction(action)
            tb.addActions(group.actions())
            tb.show()
            return tb

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
            b.SetBitmap(self.bitmap(self.modes[i].icon_file))
            b.SetSize((icon_size,icon_size))
            pos = ((i%columns)*icon_size,(i//columns)*icon_size)
            b.SetPosition(pos)

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

    def display(self, show):
        from chimerax.core import window_sys
        if window_sys == "wx":
            super().display(show)
        else:
            if show:
                f = self.buttons.show
            else:
                f = self.buttons.hide
            self.session.ui.thread_safe(f)

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, MouseModePanel, 'mouse_modes')
