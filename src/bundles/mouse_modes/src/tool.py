# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MouseModePanel(ToolInstance):

    SESSION_SKIP = True
    SESSION_ENDURING = True
    help = "help:user/tools/mousemodes.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.mouse_modes = mm = session.ui.mouse_modes
        self.button_to_bind = 'right'

        parent = session.ui.main_window

        self.modes = [m for m in mm.modes if m.icon_file]
        initial_mode = [m for m in self.modes if m.name == 'zoom'][0]

        self.buttons = self.create_buttons(self.modes, self.button_to_bind,
                                           initial_mode, parent, session)

    def create_buttons(self, modes, button_to_bind, initial_mode, parent, session):
        from PyQt5.QtWidgets import QAction, QToolBar, QActionGroup
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt, QSize
        tb = QToolBar(self.display_name, parent)
        tb.setStyleSheet('QToolBar{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; border:none;}')
        tb.setIconSize(QSize(40,40))
        parent.add_tool_bar(self, Qt.LeftToolBarArea, tb)
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

    def display(self, show):
        if show:
            f = self.buttons.show
        else:
            f = self.buttons.hide
        self.session.ui.thread_safe(f)

    def displayed(self):
        return not self.buttons.isHidden()

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, MouseModePanel, 'Mouse Modes for Right Button')
