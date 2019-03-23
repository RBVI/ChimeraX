# vi: set shiftwidth=4 expandtab:

# === UCSF ChimeraX Copyright ===
# Copyright 2019 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
TabbedToolbar is reminiscent of a Microsoft Ribbon interface.
"""

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QToolBar, QWidgetAction,
    QGridLayout, QLabel, QToolButton, QAction
)

class _Group(QWidgetAction):
    # A Group is a collection of buttons that are grouped together

    # buttons are laid out in a grid
    # row 0 is the button
    # row 1 is the group title

    def __init__(self, parent, group_title):
        super().__init__(parent)
        self._buttons = []
        self.group_title = group_title
        self.icon_size = QSize(32, 32)

    def add_button(self, title, callback, icon, description):
        button_info = (title, callback, icon, description)
        self._buttons.append(button_info)
        existing_widgets = self.createdWidgets()
        column = len(self._buttons)
        for w in existing_widgets:
            self._add_button(w, column, button_info)

    def _add_button(self, parent, column, button_info):
        (title, callback, icon, description) = button_info
        if hasattr(parent, '_title'):
            self._adjust_title(parent)

        # split title into two lines if long
        if '\n' not in title and len(title) > 6:
            words = title.split()
            if len(words) > 1:
                mid_len = int(0.4 * (sum(len(w) for w in words) + len(words) - 1))
                new_title = words[0]
                i = 1
                while i < len(words) - 1:
                    if len(new_title) >= mid_len:
                        break
                    new_title += ' ' + words[i]
                    i += 1
                new_title += '\n' + words[i]
                for i in range(i + 1, len(words)):
                    new_title += ' ' + words[i]
                title = new_title

        b = QToolButton(parent)
        b.setAutoRaise(True)
        b.setIconSize(self.icon_size)
        if icon is None:
            icon = QIcon()
            style = Qt.ToolButtonTextOnly
        else:
            # if button titles not shown use: Qt.ToolButtonIconOnly
            style = Qt.ToolButtonTextUnderIcon
        b.setToolButtonStyle(style)
        action = QAction(icon, title, parent)
        if description:
            action.setToolTip(description)
        if callback is not None:
            action.triggered.connect(callback)
        b.setDefaultAction(action)
        parent._layout.addWidget(b, 0, column, Qt.AlignTop)


    def _adjust_title(self, w):
        # Readding the widget, removes the old entry, and lets us change the span
        w._layout.addWidget(w._title, 1, 0, 1, len(self._buttons) + 1, Qt.AlignHCenter | Qt.AlignBottom)

    def createWidget(self, parent):
        w = QWidget(parent)
        w.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        layout = w._layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for column, button_info in enumerate(self._buttons):
            self._add_button(w, column, button_info)
        w._title = QLabel(self.group_title, parent)
        self._adjust_title(w)
        w.setLayout(layout)
        return w

class TabbedToolbar(QTabWidget):
    # A Microsoft Office ribbon-style interface

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # TODO: self.tabs.setMovable(True)  # and save tab order in preferences
        self._buttons = {}

    # TODO: disable/enable button, remove button

    def add_button(self, category_title, group_title, button_title, callback, icon=None, description=None):
        tab_info = self._buttons.setdefault(category_title, {})
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            tab = tab_info['__toolbar__'] = QToolBar(self)
            self.addTab(tab, category_title)
        group = tab_info.get(group_title, None)
        if group is None:
            group = tab_info[group_title] = _Group(group, group_title)
            tab.addAction(group)
            tab.addSeparator()
        group.add_button(button_title, callback, icon, description)



if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QTextEdit
    app = QApplication(sys.argv)
    app.setApplicationName("Tabbed Toolbar Demo")
    window = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    ttb = TabbedToolbar()
    layout.addWidget(ttb)
    ttb.add_button(
            'Graphics', 'Background', 'White', lambda e: print(e, 'white'),
            None, 'Set white background')
    ttb.add_button(
            'Graphics', 'Background', 'Black', lambda e: print('black'),
            None, 'Set black background')
    ttb.add_button(
            'Graphics', 'Lighting', 'Soft', lambda e: print('soft'),
            None, 'Use ambient lighting')
    ttb.add_button(
            'Graphics', 'Lighting', 'Full', lambda e: print('full'),
            None, 'Use full lighting')
    ttb.add_button(
            'Molecular Display', 'Styles', 'Sticks', lambda e: print('sticks'),
            None, 'Display atoms in stick style')
    ttb.add_button(
            'Molecular Display', 'Styles', 'Spheres', lambda e: print('spheres'),
            None, 'Display atoms in sphere style')
    ttb.add_button(
            'Molecular Display', 'Styles', 'Ball and stick', lambda e: print('bs'),
            None, 'Display atoms in ball and stick style')
    layout.addWidget(QTextEdit())
    window.setLayout(layout)
    window.show()
    sys.exit(app.exec_())
