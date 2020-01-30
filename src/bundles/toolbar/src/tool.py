# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings
from copy import deepcopy
from PyQt5.QtWidgets import QFrame, QTreeWidget, QTreeWidgetItem
from PyQt5.QtCore import pyqtSignal

defaults = {
    "home_tab": [
        # All buttons are links to existing buttons
        ("File", [
            "ChimeraX-Toolbar:Open",  # open
            "ChimeraX-Toolbar:Recent",  # recent
            "ChimeraX-Toolbar:Save",  # save
        ]),
        ("Images", [
            "ChimeraX-Shortcuts:sx",  # snapshot
            "ChimeraX-Shortcuts:vd",  # spin movie
        ]),
        ("Atoms", [
            "ChimeraX-Shortcuts:da",  # show
            "ChimeraX-Shortcuts:ha",  # hide
        ]),
        ("Cartoons", [
            "ChimeraX-Shortcuts:rb",  # show
            "ChimeraX-Shortcuts:hr",  # hide
        ]),
        ("Styles", [
            "ChimeraX-Shortcuts:st",  # stick
            "ChimeraX-Shortcuts:sp",  # sphere
            "ChimeraX-Shortcuts:bs",  # ball
        ]),
        ("Background", [
            "ChimeraX-Shortcuts:wb",  # white
            "ChimeraX-Shortcuts:bk",  # black
        ]),
        ("Lighting", [
            "ChimeraX-Shortcuts:ls",  # simple
            "ChimeraX-Shortcuts:la",  # soft
            "ChimeraX-Shortcuts:lf",  # full
        ]),
    ],
}

_settings = None
_tb = None


def get_toolbar_singleton(session, create=True):
    global _tb
    if create and _tb is None:
        _tb = ToolbarTool(session, "Toolbar")
    return _tb


class _ToolbarSettings(Settings):
    AUTO_SAVE = {
        "show_button_labels": True,
        "show_section_labels": True,
    }
    EXPLICIT_SAVE = deepcopy(defaults)


class ToolbarTool(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SAVE = False
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Toolbar"
        global _settings
        if _settings is None:
            _settings = _ToolbarSettings(self.session, "Toolbar")
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, hide_title_bar=True)
        self._build_ui()
        self.tool_window.fill_context_menu = self.fill_context_menu
        session.triggers.add_handler('set right mouse', self._set_right_mouse_button)

    def _build_ui(self):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        margins = layout.contentsMargins()
        margins.setTop(0)
        margins.setBottom(0)
        layout.setContentsMargins(margins)
        self.ttb = TabbedToolbar(
            self.tool_window.ui_area, show_section_titles=_settings.show_section_labels,
            show_button_titles=_settings.show_button_labels)
        layout.addWidget(self.ttb)
        self._build_tabs()
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.manage(self.PLACEMENT)

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from PyQt5.QtWidgets import QAction
        button_labels = QAction("Show button labels", menu)
        button_labels.setCheckable(True)
        button_labels.setChecked(_settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        section_labels = QAction("Show section labels", menu)
        section_labels.setCheckable(True)
        section_labels.setChecked(_settings.show_section_labels)
        section_labels.toggled.connect(lambda arg, f=self._set_section_labels: f(arg))
        menu.addAction(section_labels)

    def _set_button_labels(self, show_button_labels):
        _settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_section_labels(self, show_section_labels):
        _settings.show_section_labels = show_section_labels
        self.ttb.set_show_section_titles(show_section_labels)

    def build_home_tab(self):
        # (re)Build Home tab from settings
        from PyQt5.QtGui import QIcon
        self.ttb.clear_tab("Home")
        last_section = None
        for (section, compact, display_name, icon_path, description, link, bundle_info, name, kw) in _home_layout(self.session, _settings.home_tab):
            if section != last_section:
                last_section = section
                if compact:
                    self.ttb.set_section_compact("Home", section, True)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)

            def callback(event, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    "Home", section, display_name, callback,
                    icon, description, **kw)

    def _build_tabs(self):
        # add buttons from toolbar manager
        from PyQt5.QtGui import QIcon
        from .manager import fake_mouse_mode_bundle_info
        self.right_mouse_buttons = {}
        self.current_right_mouse_button = None

        self.build_home_tab()

        # Build other tabs from toolbar manager
        toolbar = self.session.toolbar._toolbar
        last_tab = None
        last_section = None
        for (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw) in _other_layout(self.session, toolbar):
            if tab != last_tab:
                last_tab = tab
                last_section = None
            if section != last_section:
                last_section = section
                if compact:
                    self.ttb.set_section_compact(tab, section, True)
            if bundle_info == fake_mouse_mode_bundle_info:
                kw["vr_mode"] = name  # Allows VR to recognize mouse mode buttons
                rmbs = self.right_mouse_buttons.setdefault(name, [])
                if icon_path is None:
                    m = self.session.ui.mouse_modes.named_mode(name)
                    if m is not None:
                        icon_path = m.icon_path
                rmbs.append((tab, section, display_name, icon_path))
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)

            def callback(event, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    tab, section, display_name, callback,
                    icon, description, **kw)
        self.ttb.show_tab('Home')
        self._set_right_mouse_button('init', self.session.ui.mouse_modes.mode("right", exact=True))

    def _set_right_mouse_button(self, trigger_name, mode):
        # highlight current right mouse button
        name = mode.name if mode is not None else None
        if name == self.current_right_mouse_button:
            return

        set_sections = set()
        has_button = name in self.right_mouse_buttons
        if has_button:
            for info in self.right_mouse_buttons[name]:
                tab_title, section_title, _, _ = info
                set_sections.add((tab_title, section_title))

        if self.current_right_mouse_button is not None:
            # remove highlighting
            for info in self.right_mouse_buttons[self.current_right_mouse_button]:
                tab_title, section_title, button_title, icon_path = info
                redo = (tab_title, section_title) not in set_sections
                self.ttb.remove_button_highlight(tab_title, section_title, button_title, redo=redo)
        if not has_button:
            return
        # highlight button(s)
        self.current_right_mouse_button = name
        for info in self.right_mouse_buttons[name]:
            tab_title, section_title, button_title, icon_path = info
            self.ttb.add_button_highlight(tab_title, section_title, button_title)


def _home_layout(session, home_tab):
    # interact through buttons in home tab
    # All buttons were vetted, so silently skip missing ones
    for section_title, buttons in _settings.home_tab:
        compact = False
        if type(section_title) is tuple:
            section_title, compact = section_title
        for link in buttons:
            kw = {}
            display_name = None
            if type(link) is tuple:
                link, display_name = link
            try:
                bundle_name, name = link.split(':', maxsplit=1)
            except ValueError:
                continue
            bi = session.toolshed.find_bundle(bundle_name, session.logger, installed=True)
            if not bi:
                continue
            pi = bi.providers.get(name, None)
            if not pi:
                continue
            pi_manager, pi_kw = pi
            if display_name is None:
                display_name = pi_kw.get("display_name", None)
                if display_name is None:
                    display_name = name
            try:
                icon_path = pi_kw["icon"]
                description = pi_kw["description"]
            except KeyError:
                continue
            if description and not description[0].isupper():
                description = description.capitalize()
            if icon_path is not None:
                icon_path = bi.get_path('icons/%s' % icon_path)
            yield (section_title, compact, display_name, icon_path, description, link, bi, name, kw)


def _other_layout(session, toolbar, hide_hidden=True):
    for tab in _layout(toolbar, "tabs"):
        if tab.startswith("__") or tab not in toolbar:
            continue
        tab_info = toolbar[tab]
        for section in _layout(tab_info, "%s sections" % tab):
            if section.startswith("__") or section not in tab_info:
                continue
            section_info = tab_info[section]
            for display_name in _layout(section_info, "%s %s buttons" % (tab, section)):
                if display_name.startswith("__") or display_name not in section_info:
                    continue
                args = section_info[display_name]
                (name, bundle_info, icon_path, description, kw) = args
                if hide_hidden and "hidden" in kw:
                    continue
                if description and not description[0].isupper():
                    description = description.capitalize()
                compact = "__compact__" in section_info
                yield (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw)


def _layout(d, what):
    # Home is always first
    if "__layout__" not in d:
        keys = [k for k in d if not k.startswith("__")]
        try:
            home = keys.index("Home")
        except ValueError:
            keys.insert(0, "Home")
        else:
            if home != 0:
                keys = ["Home"] + keys[0:home] + keys[home + 1:]
        return keys
    import copy
    layout = copy.deepcopy(d["__layout__"])
    for k in d:
        if k == "Home" or k.startswith("__"):
            continue
        if k not in layout:
            layout[k] = ["Home"]
        else:
            layout[k].add("Home")
    if "Home" in layout and layout["Home"]:
        raise RuntimeError("%s: 'Home' must be first" % what)
    layout["Home"] = []
    from chimerax.core import order_dag
    ordered = []
    try:
        for n in order_dag.order_dag(layout):
            ordered.append(n)
    except order_dag.OrderDAGError as e:
        raise RuntimeError("%s: %s" % (what, e))
    return ordered


# tree item types:
TAB_TYPE = 1
SECTION_TYPE = 2
BUTTON_TYPE = 3
GROUP_TYPE = 4


class _HomeTab(QTreeWidget):

    childDraggedAndDropped = pyqtSignal(
                QTreeWidgetItem, int, int, QTreeWidgetItem, int, name="childDraggedAndDropped")

    def __init__(self, *args, **kw):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QAbstractItemView
        super().__init__(*args, **kw)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        # disable dropping button above section
        # TODO: remove this so sections can be moved
        self.invisibleRootItem().setFlags(Qt.NoItemFlags)

    def dragEnterEvent(self, event):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidgetItemIterator
        # alter drop targets based on what is being dragged
        if event.source() != self:
            event.ignore()
            return
        kids = self.selectedItems()
        if len(kids) == 0:
            return
        source_type = kids[0].type()
        if source_type == SECTION_TYPE:
            self.invisibleRootItem().setFlags(Qt.ItemIsDropEnabled)
        else:
            self.invisibleRootItem().setFlags(Qt.NoItemFlags)

        accept_drop = source_type == BUTTON_TYPE
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if accept_drop:
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
        return super().dragEnterEvent(event)
        return
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            it += 1
            accept_drop = False
            item_type = item.type()
            if source_type == BUTTON_TYPE and item_type == SECTION_TYPE:
                accept_drop = True
            if accept_drop:
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
        return super().dragEnterEvent(event)

    def dropEvent(self, event):
        # TODO: prevent dropping button at section level
        # signal from https://vicrucann.github.io/tutorials/qtreewidget-child-drag-notify/
        from PyQt5.QtCore import Qt
        if not isinstance(event.source(), _HomeTab):
            event.ignore()
            return
        # save: event.setDropAction(Qt.MoveAction)
        # save: super().dropEvent(event)
        # row number before the drag - initial position
        kids = self.selectedItems()
        if len(kids) == 0:
            return
        start = self.indexFromItem(kids[0]).row()
        end = start  # assume only 1 kid can be dragged
        parent = kids[0].parent()

        # perform the default implementation
        super().dropEvent(event)

        # get new index
        row = self.indexFromItem(kids[0]).row
        destination = kids[0].parent()

        # emit signal about the move
        self.childDraggedAndDropped.emit(parent, start, end, destination, row)


class CustomizeTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False        # No session saving for now
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html#customize"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        tw.title = "Customize Toolbar Home Tab"
        self._build_ui()
        tw.manage(placement=None)

    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QLabel, QPushButton,
            QTreeWidget, QTreeWidgetItem,
            QGridLayout, QHBoxLayout,
        )
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt
        from .manager import fake_mouse_mode_bundle_info
        # widget layout:
        parent = self.tool_window.ui_area
        main_layout = QGridLayout()
        parent.setLayout(main_layout)
        self.instructions = QLabel(parent)
        main_layout.addWidget(self.instructions, 1, 1, 1, 2)
        self.home = _HomeTab(parent)
        main_layout.addWidget(self.home, 2, 1)
        self.home.setColumnCount(1)
        self.other = QTreeWidget(parent)
        main_layout.addWidget(self.other, 2, 2)
        self.other.setColumnCount(1)
        mod_layout = QHBoxLayout()
        main_layout.addLayout(mod_layout, 3, 1, Qt.AlignCenter)
        line = QHLine(parent)
        main_layout.addWidget(line, 4, 1, 1, 2)
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout, 5, 1, 1, 2, Qt.AlignRight)

        # TODO: group and ungroup buttons for drop downs
        new_section = QPushButton("New section", parent)
        new_section.setToolTip("Add another section to Home tab")
        new_section.clicked.connect(self.new_section)
        mod_layout.addWidget(new_section)
        remove = QPushButton("Remove", parent)
        remove.setToolTip("Remove selection items")
        remove.clicked.connect(self.remove)
        mod_layout.addWidget(remove)

        # bottom section
        save = QPushButton("Save", parent)
        save.setToolTip("Save current Home tab configuration")
        save.clicked.connect(self.save)
        bottom_layout.addWidget(save)
        reset = QPushButton("Reset", parent)
        reset.setToolTip("Reset Home tab to default configuration")
        reset.clicked.connect(self.reset)
        bottom_layout.addWidget(reset)
        restore = QPushButton("Restore", parent)
        restore.setToolTip("Restore previously saved Home tab configuration")
        restore.clicked.connect(self.restore)
        bottom_layout.addWidget(restore)
        close = QPushButton("Close", parent)
        close.setToolTip("Close dialog")
        close.clicked.connect(self.close)
        bottom_layout.addWidget(close)

        # widget contents/customization:
        self.instructions.setText("""
        <h1>Customize Toolbar Home Tab</h1>

        Instructional text on how to use inteface.
        """)

        self.build_home_tab()
        self.home.childDraggedAndDropped.connect(self.update)

        self.other.setHeaderLabels(["Available Buttons"])
        other_flags = Qt.ItemIsEnabled | Qt.ItemIsDragEnabled
        toolbar = self.session.toolbar._toolbar
        last_tab = None
        last_section = None
        tab_item = None
        section_item = None
        for (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw) in _other_layout(self.session, toolbar, hide_hidden=False):
            if bundle_info == fake_mouse_mode_bundle_info:
                continue
            if tab != last_tab:
                last_tab = tab
                last_section = None
                tab_item = QTreeWidgetItem(self.other, [tab], TAB_TYPE)
                tab_item.setFlags(Qt.ItemIsEnabled)
                self.other.expandItem(tab_item)
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(tab_item, [section], SECTION_TYPE)
                section_item.setFlags(other_flags)
                self.other.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name}"], BUTTON_TYPE)
            item.setFlags(other_flags)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)
                item.setIcon(0, icon)
            item.setToolTip(0, description)

    def build_home_tab(self):
        # the following is very similar to code for toolbar layout
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QIcon
        from PyQt5.QtWidgets import QTreeWidgetItem
        if self.home.topLevelItemCount() != 0:
            self.home.itemChanged.disconnect()
            self.home.clear()
        self.home.setHeaderLabels(["Home Tab"])
        home_buttons = (
            Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            | Qt.ItemNeverHasChildren | Qt.ItemIsDragEnabled
        )
        home_sections = (
            Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable | Qt.ItemIsDropEnabled | Qt.ItemIsDragEnabled
        )
        last_section = None
        section_item = None
        for (section, compact, display_name, icon_path, description, link, bi, name, kw) in _home_layout(self.session, _settings.home_tab):
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(self.home, [section], SECTION_TYPE)
                section_item.setFlags(home_sections)
                section_item.setCheckState(0, Qt.Checked if compact else Qt.Unchecked)
                self.home.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name}"], BUTTON_TYPE)
            item.setData(0, Qt.UserRole, link)
            item.setFlags(home_buttons)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)
                item.setIcon(0, icon)
            item.setToolTip(0, description)
        if self.home.topLevelItemCount() != 0:
            self.home.itemChanged.connect(self.update)

    def update(self, *args):
        # propagate user changes to home tab
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidgetItemIterator
        home_tab = []
        cur_section = []
        it = QTreeWidgetItemIterator(self.home)
        while it.value():
            item = it.value()
            it += 1
            if item.type() == BUTTON_TYPE:
                display_name = item.text(0)
                link = item.data(0, Qt.UserRole)
                # TODO: examine linked item to see if display_name is the same
                name = link.split(sep=':', maxsplit=1)[1]
                if name == display_name:
                    cur_section.append(link)
                else:
                    cur_section.append((link, display_name))
            elif item.type() == SECTION_TYPE:
                name = item.text(0)
                cur_section = []
                if item.checkState(0):
                    home_tab.append(((name, True), cur_section))
                else:
                    home_tab.append((name, cur_section))
        _settings.home_tab = home_tab
        tb = get_toolbar_singleton(self.session, create=False)
        if tb:
            tb.build_home_tab()

    def update_from_settings(self):
        self.build_home_tab()
        tb = get_toolbar_singleton(self.session, create=False)
        if tb:
            tb.build_home_tab()

    def new_section(self):
        # add new section to home tab
        pass

    def remove(self):
        # remove selected sections/buttons from home tab
        import sys
        print([si.text(0) for si in self.home.selectedItems()], file=sys.__stderr__)
        # TODO:

    def save(self):
        # save current configuration in preferences
        _settings.save()

    def reset(self):
        # reset current configuration in original defaults
        _settings.reset()
        self.update_from_settings()

    def restore(self):
        # restore current configuration from saved preferences
        _settings.restore()
        self.update_from_settings()

    def close(self):
        self.delete()

# Adapted QHLine from
# https://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt


class QHLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
