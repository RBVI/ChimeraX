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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QToolBar, QWidgetAction,
    QGridLayout, QLabel, QToolButton, QAction
)

_debug = False   # DEBUG


class _Group(QWidgetAction):
    # A Group is a collection of buttons that are grouped together

    # Buttons are laid out in a grid.
    # If compact, the buttons are laid out vertically with rows
    # 0 to n - 1 being the buttons and row n being the group title.
    # If not compact, then row 0 is the button and row 1 is the group title.

    # TODO: support more than 3 compact buttons by using multiple columns
    # and span the title

    def __init__(self, parent, group_title, compact, show_group_titles, show_button_titles):
        super().__init__(parent)
        self._buttons = []
        self.group_title = group_title
        self.compact = compact
        self.show_group_titles = show_group_titles
        self.show_button_titles = show_button_titles

    def add_button(self, title, callback, icon, description):
        button_info = (title, callback, icon, description)
        self._buttons.append(button_info)
        existing_widgets = self.createdWidgets()
        ordinal = len(self._buttons)
        for w in existing_widgets:
            self._add_button(w, ordinal, button_info)

    def _add_button(self, parent, ordinal, button_info):
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
        if icon is None:
            icon = QIcon()
            style = Qt.ToolButtonTextOnly
        else:
            if not self.show_button_titles:
                style = Qt.ToolButtonIconOnly
            elif self.compact:
                style = Qt.ToolButtonTextBesideIcon
            else:
                style = Qt.ToolButtonTextUnderIcon
        b.setToolButtonStyle(style)
        action = QAction(icon, title, parent)
        if description:
            action.setToolTip(description)
        if callback is not None:
            action.triggered.connect(callback)
        b.setDefaultAction(action)
        # print('Font height:', b.fondMetrics().height())  # DEBUG
        # print('Font size:', b.fontInfo().pixelSize())  # DEBUG
        # print('Icon size:', b.iconSize())  # DEBUG
        if self.compact:
            parent._layout.addWidget(b, ordinal, 0)
        else:
            if self.show_button_titles:
                b.setIconSize(2 * b.iconSize())
            else:
                b.setIconSize(3 * b.iconSize())
            parent._layout.addWidget(b, 0, ordinal, Qt.AlignTop)
        global _debug
        if _debug:
            _debug = False
            policy = b.sizePolicy()
            print('expanding:', int(policy.expandingDirections()))
            print('horizontal policy:', policy.horizontalPolicy())
            print('horizontal stretch:', policy.horizontalStretch())
            print('vertical policy:', policy.verticalPolicy())
            print('vertical stretch:', policy.verticalStretch())

    def _adjust_title(self, w):
        # Readding the widget, removes the old entry, and lets us change the parameters
        if self.compact:
            w._layout.addWidget(w._title, len(self._buttons) + 1, 0, Qt.AlignHCenter | Qt.AlignBottom)
        else:
            w._layout.addWidget(w._title, 1, 0, 1, len(self._buttons) + 1, Qt.AlignHCenter | Qt.AlignBottom)

    def createWidget(self, parent):
        w = QWidget(parent)
        w.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        layout = w._layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout_buttons(w)
        w.setLayout(layout)
        return w

    def _layout_buttons(self, w):
        for column, button_info in enumerate(self._buttons):
            self._add_button(w, column, button_info)
        if self.show_group_titles:
            w._title = QLabel(self.group_title, w)
            self._adjust_title(w)

    def _clear_layout(self, layout):
        # remove/delete members of layout
        while layout.count():
            w = layout.takeAt(0).widget()
            layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()

    def set_show_group_titles(self, on_off):
        if self.show_group_titles == on_off:
            return
        self.show_group_titles = on_off
        for w in self.createdWidgets():
            if hasattr(w, '_title'):
                del w._title
            self._clear_layout(w._layout)
            self._layout_buttons(w)

    def set_show_button_titles(self, on_off):
        if self.show_button_titles == on_off:
            return
        self.show_button_titles = on_off
        for w in self.createdWidgets():
            if hasattr(w, '_title'):
                del w._title
            self._clear_layout(w._layout)
            self._layout_buttons(w)


class TabbedToolbar(QTabWidget):
    # A Microsoft Office ribbon-style interface

    def __init__(self, *args, show_group_titles=True, show_button_titles=True, **kw):
        super().__init__(*args, **kw)
        # TODO: self.tabs.setMovable(True)  # and save tab order in preferences
        self._buttons = {}
        self.show_group_titles = show_group_titles
        self.show_button_titles = show_button_titles

    # TODO: disable/enable button/group, remove button

    def add_button(self, category_title, group_title, button_title, callback, icon=None, description=None, compact=False):
        tab_info = self._buttons.setdefault(category_title, {})
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            tab = tab_info['__toolbar__'] = QToolBar(self)
            self.addTab(tab, category_title)
        group = tab_info.get(group_title, None)
        if group is None:
            group = tab_info[group_title] = _Group(
                group, group_title, compact, self.show_group_titles, self.show_button_titles)
            tab.addAction(group)
            tab.addSeparator()
        group.add_button(button_title, callback, icon, description)

    def show_category(self, category_title):
        tab_info = self._buttons.get(category_title, None)
        if tab_info is None:
            return
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            return
        index = self.indexOf(tab)
        if index == -1:
            return
        self.setCurrentIndex(index)

    def _recompute_tab_sizes(self):
        # can't shrink vertically unless the size of all tabs are recomputed
        # (since Qt delays recomputing a tab's size until it is visible)
        current = self.currentIndex()
        for i in range(self.count()):
            if i == current:
                # Qt already handles current tab
                continue
            tab = self.widget(i)
            tab.adjustSize()
            tab.updateGeometry()  # TODO: not needed all of the time

    def set_show_group_titles(self, on_off):
        if self.show_group_titles == on_off:
            return
        self.show_group_titles = on_off
        for category_title, tab_info in self._buttons.items():
            for group_title, group in tab_info.items():
                if group_title == "__toolbar__":
                    continue
                group.set_show_group_titles(on_off)
        if not on_off:
            self._recompute_tab_sizes()

    def set_show_button_titles(self, on_off):
        if self.show_button_titles == on_off:
            return
        self.show_button_titles = on_off
        for category_title, tab_info in self._buttons.items():
            for group_title, group in tab_info.items():
                if group_title == "__toolbar__":
                    continue
                group.set_show_button_titles(on_off)
        if not on_off:
            self._recompute_tab_sizes()


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
