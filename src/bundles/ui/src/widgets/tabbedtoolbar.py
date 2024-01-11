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

Note: This widget is under active development and the API may change.

TODO: documnentation!
"""

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QWidget, QTabWidget, QToolBar, QWidgetAction,
    QGridLayout, QLabel, QToolButton
)
from Qt.QtGui import QPainter, QIcon, QColor, QPixmap, QAction

_debug = False   # DEBUG


def split_title(title):
    """Try to split title into two lines"""
    words = title.split()
    if len(words) <= 1:
        return title
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
    return new_title


class _ButtonInfo:

    __slots__ = (
        "title", "callback", "icon", "description", "group", "vr_mode", "highlight_icon",
        "enabled"
    )

    def __init__(self, title, callback, icon, description, group, vr_mode, enabled):
        self.title = title
        self.callback = callback
        self.icon = icon
        self.description = description
        self.group = group
        self.vr_mode = vr_mode
        self.enabled = enabled
        self.highlight_icon = None


class _Section(QWidgetAction):
    # A Section is a collection of buttons that are adjacent to each other

    # Buttons are laid out in a grid.
    # If compact, the buttons are laid out vertically with rows
    # 0 to n - 1 being the buttons and row n being the section title.
    # If not compact, then row 0 is the button and row 1 is the section title.

    def __init__(self, parent, section_title, show_section_titles, show_button_titles, highlight_color):
        super().__init__(parent)
        self._buttons = []
        self._groups = {}   # { toolbar-widget: { group-name: qtoolbutton } }
        self._actions = {}  # keep references to actions in menus { button-title: [action] }
        self.section_title = section_title
        self.show_section_titles = show_section_titles
        self.show_button_titles = show_button_titles
        self.highlight_color = highlight_color
        self.compact = False
        if show_button_titles:
            self.compact_height = 3
        else:
            self.compact_height = 2

    def add_button(self, title, callback, icon, description, group, vr_mode, enabled):
        if group and self.compact:
            raise ValueError("Can not use grouped buttons in a compact section")
        index = len(self._buttons)
        button_info = _ButtonInfo(title, callback, icon, description, group, vr_mode, enabled)
        self._buttons.append(button_info)
        existing_widgets = self.createdWidgets()
        for w in existing_widgets:
            self._add_button(w, index, button_info)

    def _add_button(self, parent, index, button_info):
        if hasattr(parent, '_title'):
            self._adjust_title(parent)

        # split title into two lines if long
        orig_title = button_info.title
        title = orig_title
        if '\n' not in title and len(title) > 6:
            title = split_title(title)

        if button_info.highlight_icon is None:
            icon = button_info.icon
        else:
            icon = button_info.highlight_icon

        if not button_info.group:
            group_first = group_follow = False
        else:
            buttons = self._groups.setdefault(parent, {})
            group_first = button_info.group not in buttons  # first button in drop down
            group_follow = not group_first                  # subsequent buttons
        if not group_follow:
            b = QToolButton(parent)
            if button_info.vr_mode is not None:
                b.vr_mode = button_info.vr_mode
            b.setAutoRaise(True)
            if icon is None:
                style = Qt.ToolButtonStyle.ToolButtonTextOnly
            else:
                if not self.show_button_titles:
                    style = Qt.ToolButtonStyle.ToolButtonIconOnly
                elif self.compact:
                    style = Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                else:
                    style = Qt.ToolButtonStyle.ToolButtonTextUnderIcon
            b.setToolButtonStyle(style)
        if icon is None:
            action = QAction(title)
        else:
            action = QAction(icon, title)
        if button_info.description:
            action.setToolTip(button_info.description)
        if button_info.callback is not None:
            action.triggered.connect(button_info.callback)
        if not button_info.enabled:
            action.setEnabled(False)
        actions = self._actions.setdefault(orig_title, {})
        actions[parent] = action
        if group_follow:
            button = self._groups[parent][button_info.group]
            button.addAction(action)
        else:
            if not group_first:
                b.setDefaultAction(action)
            else:
                b.setPopupMode(b.ToolButtonPopupMode.MenuButtonPopup)
                b.triggered.connect(lambda action, b=b: self._update_button_action(b, action))
                self._groups[parent][button_info.group] = b
                b.addAction(action)
                self._update_button_action(b, action)

        # print('Font height:', b.fontMetrics().height())  # DEBUG
        # print('Font size:', b.fontInfo().pixelSize())  # DEBUG
        # print('Icon size:', b.iconSize())  # DEBUG
        if not group_follow:
            if self.compact:
                row = index % self.compact_height
                if row < self.compact_height:
                    parent._layout.setRowStretch(row, 1)
                column = index // self.compact_height
                parent._layout.addWidget(b, row, column, Qt.AlignLeft | Qt.AlignVCenter)
            else:
                if not self.show_button_titles or button_info.icon is None:
                    align = Qt.AlignCenter
                else:
                    align = Qt.AlignTop
                b.setIconSize(2 * b.iconSize())
                parent._layout.addWidget(b, 0, index, align)
        global _debug
        if _debug:
            _debug = False
            policy = b.sizePolicy()
            print('expanding:', int(policy.expandingDirections()))
            print('horizontal policy:', policy.horizontalPolicy())
            print('horizontal stretch:', policy.horizontalStretch())
            print('vertical policy:', policy.verticalPolicy())
            print('vertical stretch:', policy.verticalStretch())

    def _update_button_action(self, button, action):
        button.setDefaultAction(action)
        lines = button.text().split('\n')
        fm = button.fontMetrics()
        width = max(fm.horizontalAdvance(text) for text in lines)
        # 20 is width of arrow on right side of button
        button.setMinimumWidth(width + 20)

    def _adjust_title(self, w):
        # Readding the widget, removes the old entry, and lets us change the parameters
        size = len(self._buttons)
        if self.compact:
            span = (size + self.compact_height - 1) // self.compact_height
            w._layout.addWidget(w._title, self.compact_height, 0, 1, span,
                                Qt.AlignHCenter | Qt.AlignBottom)
        else:
            w._layout.addWidget(w._title, 1, 0, 1, size,
                                Qt.AlignHCenter | Qt.AlignBottom)

    def createWidget(self, parent):
        w = QWidget(parent)
        w.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        layout = w._layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if not self.compact:
            layout.setRowStretch(0, 1)
        self._layout_buttons(w)
        w.setLayout(layout)
        return w

    def _layout_buttons(self, w):
        for column, button_info in enumerate(self._buttons):
            self._add_button(w, column, button_info)
        if self.show_section_titles:
            w._title = QLabel(self.section_title, w)
            self._adjust_title(w)

    def _clear_layout(self, layout):
        # remove/delete members of layout
        while layout.count():
            w = layout.takeAt(0).widget()
            layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()

    def _redo_layout(self):
        self._actions.clear()
        # TODO: destroy group menus
        self._groups.clear()
        for w in self.createdWidgets():
            if hasattr(w, '_title'):
                del w._title
            self._clear_layout(w._layout)
            self._layout_buttons(w)
            w.updateGeometry()
            w.adjustSize()

    def set_compact(self, on_off):
        if self.compact == on_off:
            return
        for button_info in self._buttons:
            if button_info.group:
                raise ValueError("Can not make a section compact that has grouped buttons")
        self.compact = on_off
        self._redo_layout()

    def set_show_section_titles(self, on_off):
        if self.show_section_titles == on_off:
            return
        self.show_section_titles = on_off
        self._redo_layout()

    def set_show_button_titles(self, on_off):
        if self.show_button_titles == on_off:
            return
        self.show_button_titles = on_off
        if on_off:
            self.compact_height = 3
        else:
            self.compact_height = 2
        self._redo_layout()

    def get_qt_button_action(self, parent, title):
        actions = self._actions.get(title, None)
        if actions:
            return actions.get(parent)
        return None

    def add_button_highlight(self, title, redo=True):
        for button_info in self._buttons:
            if button_info.title == title:
                break
        else:
            return
        icon = button_info.icon
        if icon is None or icon.isNull():
            # Make a single color icon
            pm = QPixmap(256, 256)
            pm.fill(self.highlight_color)
        else:
            sizes = icon.availableSizes()
            sizes.sort(key=lambda s: s.width())
            pm = icon.pixmap(icon.actualSize(sizes[-1]))
#            with QPainter(pm) as p:
            p = None
            try:
                p = QPainter(pm)
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
                if 1:
                    # draw filled
                    p.fillRect(pm.rect(), self.highlight_color)
                else:
                    # draw outline
                    r = pm.rect()
                    pen_width = max(r.width(), r.height()) / 16
                    p.setPen(self.highlight_color)
                    pen = p.pen()
                    pen.setWidth(pen_width)
                    p.setPen(pen)
                    adj = pen_width / 2
                    p.drawRect(r.adjusted(adj, adj, -adj, -adj))
            finally:
                if p:
                    p.end()
        button_info.highlight_icon = QIcon(pm)
        if redo:
            self._redo_layout()

    def remove_button_highlight(self, title, redo=True):
        for button_info in self._buttons:
            if button_info.title == title:
                break
        else:
            return
        button_info.highlight_icon = None
        if redo:
            self._redo_layout()

    def set_highlight_color(self, color):
        self.highlight_color = color
        for button_info in self._buttons:
            if button_info.highlight_icon is None:
                continue
            self.add_button_highlight(button_info.title, redo=False)
        self._redo_layout()

    def set_enabled(self, enabled, button_title, redo=True):
        for button_info in self._buttons:
            if button_info.title == button_title:
                break
        else:
            raise ValueError(f"Didn't find button '{button_title}'")
        if button_info.enabled == enabled:
            return
        button_info.enabled = enabled
        if redo:
            existing_widgets = self.createdWidgets()
            for parent in existing_widgets:
                action = self.get_qt_button_action(parent, button_title)
                if action is None:
                    continue
                action.setEnabled(enabled)

    def show_group_button(self, button_title):
        for button_info in self._buttons:
            if button_info.title == button_title:
                break
        else:
            return
        group = button_info.group
        if group is None:
            return
        existing_widgets = self.createdWidgets()
        for parent in existing_widgets:
            action = self.get_qt_button_action(parent, button_title)
            if action is None:
                continue
            b = self._groups[parent][group]
            b.setDefaultAction(action)


class TabbedToolbar(QTabWidget):
    # A Microsoft Office ribbon-style interface

    def __init__(self, *args, show_section_titles=True, show_button_titles=True, **kw):
        super().__init__(*args, **kw)
        # TODO: self.tabs.setMovable(True)  # and save tab order in preferences
        self._buttons = {}  # { tab_title: { section_title: _Section() } }
        self.show_section_titles = show_section_titles
        self.show_button_titles = show_button_titles
        # self.setStyleSheet("* { padding: 0; margin: 0; border: 1px inset red; } *::separator { background-color: green; width: 1px; }")
        # self.setStyleSheet("* { padding: 0; margin: 0; } *::separator { width: 1px; }")
        self.setStyleSheet("* { padding: 0; margin: 0; }")
        # self.setStyleSheet("*::separator { width: 1px; }")
        self._highlight_color = QColor("light green")

    # TODO: disable/enable button/section, remove button

    def _get_section(self, tab_title, section_title, create=True):
        tab_info = self._buttons.setdefault(tab_title, {})
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            if not create:
                return None
            tab = tab_info['__toolbar__'] = QToolBar(self)
            self.addTab(tab, tab_title)
        section = tab_info.get(section_title, None)
        if section is None:
            if not create:
                return None
            section = tab_info[section_title] = _Section(
                section, section_title, self.show_section_titles, self.show_button_titles,
                self._highlight_color)
            tab.addAction(section)
            tab.addSeparator()
        return section

    def set_section_compact(self, tab_title, section_title, on_off):
        section = self._get_section(tab_title, section_title)
        section.set_compact(on_off)

    def add_button(self, tab_title, section_title, button_title, callback, icon=None, description=None, *, group=None, vr_mode=None, enabled=True):
        if isinstance(enabled, str):
            enabled = enabled not in ('False', 'false', '0', 'off')
        section = self._get_section(tab_title, section_title)
        section.add_button(button_title, callback, icon, description, group, vr_mode, enabled)

    def show_tab(self, tab_title):
        """Make given tab the current tab"""
        tab_info = self._buttons.get(tab_title, None)
        if tab_info is None:
            return
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            return
        index = self.indexOf(tab)
        if index == -1:
            return
        self.setCurrentIndex(index)

    def clear_tab(self, tab_title):
        """Clear contents of tab"""
        tab_info = self._buttons.get(tab_title, None)
        if tab_info is None:
            return
        tab = tab_info.get("__toolbar__", None)
        if tab is None:
            return
        tab.clear()
        self._buttons[tab_title].clear()
        self._buttons[tab_title]["__toolbar__"] = tab

    def _recompute_tab_sizes(self):
        # can't shrink vertically unless the size of all tabs are recomputed
        # (since Qt delays recomputing a tab's size until it is visible)
        current = self.currentIndex()
        for i in range(self.count()):
            if i == current:
                # Qt already handles current tab
                continue
            tab = self.widget(i)
            tab.updateGeometry()  # TODO: not needed all of the time
            tab.adjustSize()

    def set_show_section_titles(self, on_off):
        if self.show_section_titles == on_off:
            return
        self.show_section_titles = on_off
        for tab_title, tab_info in self._buttons.items():
            for section_title, section in tab_info.items():
                if section_title == "__toolbar__":
                    continue
                section.set_show_section_titles(on_off)
        if not on_off:
            self._recompute_tab_sizes()

    def set_show_button_titles(self, on_off):
        if self.show_button_titles == on_off:
            return
        self.show_button_titles = on_off
        for tab_title, tab_info in self._buttons.items():
            for section_title, section in tab_info.items():
                if section_title == "__toolbar__":
                    continue
                section.set_show_button_titles(on_off)
        if not on_off:
            self._recompute_tab_sizes()

    def add_button_highlight(self, tab_title, section_title, button_title, *, redo=True):
        section = self._get_section(tab_title, section_title, create=False)
        if section is None:
            return
        self._hide_toolbar_rollover(tab_title)  # Work around ChimeraX bug #3152
        section.add_button_highlight(button_title, redo=redo)

    def remove_button_highlight(self, tab_title, section_title, button_title, *, redo=True):
        section = self._get_section(tab_title, section_title, create=False)
        if section is None:
            return
        self._hide_toolbar_rollover(tab_title)  # Work around ChimeraX bug #3152
        section.remove_button_highlight(button_title, redo=redo)

    def set_enabled(self, enabled, tab_title, section_title, button_title):
        section = self._get_section(tab_title, section_title, create=False)
        if section is None:
            raise ValueError(f"Didn't find section '{section_title}' in tab '{tab_title}'")
        section.set_enabled(enabled, button_title)

    def _hide_toolbar_rollover(self, tab_title):
        # Hack to hide toolbar rollover menu if it is shown.
        # Needed to work around ChimeraX bug #3152.
        tab_info = self._buttons.get(tab_title)
        if tab_info:
            toolbar = tab_info.get("__toolbar__", None)
            if toolbar:
                for w in toolbar.children():
                    if isinstance(w, QToolButton):
                        menu = w.menu()
                        if menu and menu.isVisible():
                            menu.hide()
                            return True
        return False

    def set_highlight_color(self, qcolor):
        if qcolor == self._highlight_color:
            return
        self._highlight_color = qcolor
        for tab_title, tab_info in self._buttons.items():
            for section_title, section in tab_info.items():
                if section_title == "__toolbar__":
                    continue
                section.set_highlight_color(qcolor)

    def show_group_button(self, tab_title, section_title, button_title):
        section = self._get_section(tab_title, section_title, create=False)
        if section is None:
            raise ValueError(f"Didn't find section '{section_title}' in tab '{tab_title}'")
        section.show_group_button(button_title)

if __name__ == "__main__":
    import sys
    from Qt.QtWidgets import QApplication, QVBoxLayout, QTextEdit
    app = QApplication(sys.argv)
    app.setApplicationName("Tabbed Toolbar Demo")
    window = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    ttb = TabbedToolbar()
    layout.addWidget(ttb)
    ttb.add_button(
        'Graphics', 'Background', 'White', lambda: print('white'),
        None, 'Set white background')
    ttb.add_button(
        'Graphics', 'Background', 'Black', lambda: print('black'),
        None, 'Set black background')
    ttb.add_button(
        'Graphics', 'Lighting', 'Soft', lambda: print('soft'),
        None, 'Use ambient lighting')
    ttb.add_button(
        'Graphics', 'Lighting', 'Full', lambda: print('full'),
        None, 'Use full lighting')
    ttb.add_button(
        'Molecular Display', 'Styles', 'Sticks', lambda: print('sticks'),
        None, 'Display atoms in stick style')
    ttb.add_button(
        'Molecular Display', 'Styles', 'Spheres', lambda: print('spheres'),
        None, 'Display atoms in sphere style')
    ttb.add_button(
        'Molecular Display', 'Styles', 'Ball and stick', lambda: print('bs'),
        None, 'Display atoms in ball and stick style')
    layout.addWidget(QTextEdit())
    window.setLayout(layout)
    window.show()
    sys.exit(app.exec())
