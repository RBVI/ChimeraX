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

    SESSION_ENDURING = True
    help = "help:user/tools/mousemodes.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.mouse_modes = mm = session.ui.mouse_modes
        self.button_to_bind = 'right'
        self._icon_size = 40
        self._icons_per_row = 13
        self.tool_window = None
        
        self.modes = [m for m in mm.modes if m.icon_file]
        # there should be a generic ordering scheme, but for now move distance mode
        # and bond-rotation mode up some
        mode_names = [m.name for m in self.modes]
        if "distance" in mode_names:
            index = mode_names.index("distance")
            if index > 8:
                self.modes = self.modes[:9] + [self.modes[index]] + self.modes[9:index] \
                    + self.modes[index+1:]
                mode_names = mode_names[:9] + [mode_names[index]] + mode_names[9:index] \
                    + mode_names[index+1:]
        if "bond rotation" in mode_names:
            index = mode_names.index("bond rotation")
            if index > 12:
                self.modes = self.modes[:12] + [self.modes[index]] + self.modes[12:index] \
                    + self.modes[index+1:]

        parent = session.ui.main_window
        self.buttons = self.create_toolbar(parent)
        
    def create_toolbar(self, parent):
        from PyQt5.QtWidgets import QAction, QToolBar, QActionGroup
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt, QSize
        tb = QToolBar(self.display_name, parent)
        tb.setStyleSheet('QToolBar{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; border:none;}')
        s = self._icon_size
        tb.setIconSize(QSize(s,s))
        parent.add_tool_bar(self, Qt.LeftToolBarArea, tb)
        group = QActionGroup(tb)
        for mode in self.modes:
            action = QAction(self._icon(mode.icon_file), mode.name, group)
            action.setCheckable(True)
            def button_press_cb(event, mode=mode):
                mname = mode.name
                if ' ' in mname:
                    mname = '"%s"' % mname
                from chimerax.core.commands import run
                run(self.session, 'ui mousemode %s %s' % (self.button_to_bind, mname))
            action.triggered.connect(button_press_cb)
            action.mouse_mode = mode
            group.addAction(action)
        tb.addActions(group.actions())
        tb.show()
        return tb

    def create_button_panel(self):
        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        p = tw.ui_area
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(p)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        b = self.create_panel_buttons(p)
        p.setLayout(layout)
        layout.addWidget(b)
        tw.manage(placement="side")

    def create_panel_buttons(self, parent):
        from PyQt5.QtWidgets import QAction, QFrame, QGridLayout, QToolButton, QActionGroup
        from PyQt5.QtCore import Qt, QSize
        tb = QFrame(parent)
        layout = QGridLayout(tb)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tb.setStyleSheet('QFrame{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; border:none;}')
        group = QActionGroup(tb)
        s = self._icon_size
        columns = self._icons_per_row
        for mnum,mode in enumerate(self.modes):
            b = QToolButton(tb)
            b.setIconSize(QSize(s,s))
            action = QAction(self._icon(mode.icon_file), mode.name, group)
            b.setDefaultAction(action)
            action.setCheckable(True)
            def button_press_cb(event, mode=mode):
                mname = mode.name
                if ' ' in mname:
                    mname = '"%s"' % mname
                from chimerax.core.commands import run
                run(self.session, 'ui mousemode %s %s' % (self.button_to_bind, mname))
            action.triggered.connect(button_press_cb)
            action.mouse_mode = mode
            group.addAction(action)
            row, column = mnum//columns, mnum%columns
            layout.addWidget(b, row, column)
        return tb

    def _icon(self, file):
        from os import path
        if path.isabs(file):
            p = file
        else:
            icon_dir = path.join(path.dirname(__file__), 'icons')
            p = path.join(icon_dir, file)
        from PyQt5.QtGui import QIcon
        qi = QIcon(p)
        return qi

    def display(self, show):
        if show:
            f = self.buttons.show
        else:
            f = self.buttons.hide
        self.session.ui.thread_safe(f)

    def displayed(self):
        return not self.buttons.isHidden()
        
    def display_panel(self, show):
        tw = self.tool_window
        if show:
            if tw is None:
                self.create_button_panel()
            self.tool_window.shown = True
        elif tw:
            tw.shown = False

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, MouseModePanel, 'Mouse Modes for Right Button')
