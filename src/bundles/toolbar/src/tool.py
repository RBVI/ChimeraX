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
from Qt.QtWidgets import QFrame, QTreeWidget, QTreeWidgetItem
from Qt.QtCore import Qt, Signal

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
        ("Selection", [
            "ChimeraX-SelInspector:selection inspector",  # inspect selection
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
    help = "help:user/tools/toolbar.html"  # Let ChimeraX know about our help page

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
        session.triggers.add_handler('set mouse mode', self._set_right_mouse_button)

    def _build_ui(self):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        from Qt.QtWidgets import QVBoxLayout
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
        from Qt.QtGui import QAction
        button_labels = QAction("Show button labels", menu)
        button_labels.setCheckable(True)
        button_labels.setChecked(_settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, *, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        section_labels = QAction("Show section labels", menu)
        section_labels.setCheckable(True)
        section_labels.setChecked(_settings.show_section_labels)
        section_labels.toggled.connect(lambda arg, *, f=self._set_section_labels: f(arg))
        menu.addAction(section_labels)
        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(lambda arg: self.show_settings())
        menu.addAction(settings_action)

    def show_settings(self):
        if not hasattr(self, "settings_tool"):
            self.settings_tool = ToolbarSettingsTool(
                self.session, self,
                self.tool_window.create_child_window("Toolbar Settings", close_destroys=False))
            self.settings_tool.tool_window.manage(None)
        self.settings_tool.tool_window.shown = True

    def _set_button_labels(self, show_button_labels):
        _settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_section_labels(self, show_section_labels):
        _settings.show_section_labels = show_section_labels
        self.ttb.set_show_section_titles(show_section_labels)

    def build_home_tab(self):
        # (re)Build Home tab from settings
        from Qt.QtGui import QIcon
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

            def callback(*, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    "Home", section, display_name, callback,
                    icon, description, **kw)

    def _build_tabs(self):
        # add buttons from toolbar manager
        from Qt.QtGui import QIcon
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

            def callback(*, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    tab, section, display_name, callback,
                    icon, description, **kw)
        self.ttb.show_tab('Home')
        self._set_right_mouse_button('init', self.session.ui.mouse_modes.mode("right", exact=True))

    def _set_right_mouse_button(self, trigger_name, data):
        # highlight current right mouse button
        if trigger_name == 'init':
            mode = data
        else:
            button, modifiers, mode = data
            if button != "right" or modifiers:
                return
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

    def set_enabled(self, enabled, tab_title, section_title, button_title):
        self.ttb.set_enabled(enabled, tab_title, section_title, button_title)


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
            pi = bi.providers.get('toolbar/' + name, None)
            if not pi:
                continue
            pi_kw = pi
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
    from chimerax.core import is_daily_build
    if is_daily_build():
        import sys
        for key, values in layout.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                if value not in layout:
                    print(f"developer warning: toolbar '{key}' depends on non-existent '{value}'", file=sys.__stderr__)
    from chimerax.core import order_dag
    ordered = []
    try:
        for n in order_dag.order_dag(layout):
            ordered.append(n)
    except order_dag.OrderDAGError as e:
        raise RuntimeError("%s: %s" % (what, e))
    return ordered


# tree item data roles:
LINK_ROLE = Qt.ItemDataRole.UserRole
ITEM_TYPE_ROLE = Qt.ItemDataRole.UserRole + 1
# tree item types:
TAB_TYPE = 1
SECTION_TYPE = 2
BUTTON_TYPE = 3
GROUP_TYPE = 4
# tree item flags:
BUTTON_FLAGS = (
    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
    | Qt.ItemFlag.ItemNeverHasChildren | Qt.ItemFlag.ItemIsDragEnabled
)
SECTION_FLAGS = (
    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
    | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled
)


class _HomeTab(QTreeWidget):

    childDraggedAndDropped = Signal(name="childDraggedAndDropped")

    def __init__(self, parent, *args, sources=[], **kw):
        from Qt.QtCore import Qt
        from Qt.QtWidgets import QAbstractItemView
        super().__init__(*args, **kw)
        self.sources = sources
        self.setColumnCount(1)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def dragEnterEvent(self, event):
        # alter drop targets based on what is being dragged
        from Qt.QtCore import Qt
        from Qt.QtWidgets import QTreeWidgetItemIterator
        source = event.source()
        if source != self and source not in self.sources:
            event.ignore()
            return
        selected = source.selectedItems()
        if len(selected) == 0:
            event.ignore()
            return
        source_type = selected[0].data(0, ITEM_TYPE_ROLE)
        self.invisibleRootItem().setFlags(
                Qt.ItemIsDropEnabled if source_type == SECTION_TYPE else Qt.NoItemFlags)

        accept_drop = source_type == BUTTON_TYPE
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if accept_drop:
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
        return super().dragEnterEvent(event)
        # alternate implementation
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            it += 1
            accept_drop = False
            item_type = item.data(0, ITEM_TYPE_ROLE)
            if source_type == BUTTON_TYPE and item_type == SECTION_TYPE:
                accept_drop = True
            if accept_drop:
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
        return super().dragEnterEvent(event)

    def dropEvent(self, event):
        source = event.source()
        # from dragEnterEvent, we know there is at least one selected item
        original = source.selectedItems()[0]
        original_type = original.data(0, ITEM_TYPE_ROLE)
        if source == self:
            copy_subtree = False
        else:
            copy_subtree = original_type == SECTION_TYPE
        result = super().dropEvent(event)
        if copy_subtree:
            # find where it was copied to
            new_section = self.itemAt(event.pos())
            if new_section is None:
                # assume dropped below bottom
                new_section = self.topLevelItem(self.topLevelItemCount() - 1)
            parent = new_section.parent()
            if parent is None:
                parent = self.invisibleRootItem()
            if new_section.childCount() != 0:
                i = parent.indexOfChild(new_section)
                new_section = parent.child(i + 1)
                # assert new_section.childCount() == 0
            new_section.setFlags(SECTION_FLAGS)
            self.expandItem(new_section)
            for i in range(original.childCount()):
                new_child = original.child(i).clone()
                new_child.setFlags(BUTTON_FLAGS)
                new_section.addChild(new_child)
            # make sure section name is unique
            from collections import Counter
            section_name = original.text(0)
            current_section_names = []
            for i in range(parent.childCount()):
                item_name = parent.child(i).text(0)
                current_section_names.append(item_name)
            current_sections = Counter(current_section_names)
            if current_sections[section_name] > 1:
                from itertools import chain, count
                for suffix in chain(("",), count(2)):
                    new_name = f"new {section_name}{suffix}"
                    if new_name not in current_sections:
                        new_section.setText(0, new_name)
                        break
        elif source == self:
            if original_type == SECTION_TYPE:
                self.expandItem(original)
        else:
            new_button = self.itemAt(event.pos())
            new_button.setFlags(BUTTON_FLAGS)
        self.childDraggedAndDropped.emit()
        return result


class ToolbarSettingsTool:

    help_url = "help:user/tools/toolbar.html#settings"

    def __init__(self, session, toolbar, tool_window):
        self.session = session
        self.toolbar = toolbar
        self.tool_window = tool_window
        self._build_ui()

    def _build_ui(self):
        from Qt.QtWidgets import (
            QLabel, QPushButton,
            QTreeWidget, QAbstractItemView,
            QGridLayout, QHBoxLayout,
        )
        from Qt.QtGui import QIcon
        from Qt.QtCore import Qt
        from .manager import fake_mouse_mode_bundle_info
        parent = self.tool_window.ui_area
        # main widgets
        self.instructions = QLabel(parent)
        self.other = QTreeWidget(parent)
        self.home = _HomeTab(parent, sources=[self.other])
        line = QHLine(parent)
        # widget layout:
        main_layout = QGridLayout()
        parent.setLayout(main_layout)
        main_layout.addWidget(self.instructions, 1, 1, 1, 2)
        main_layout.addWidget(self.home, 2, 1)
        main_layout.addWidget(self.other, 2, 2)
        mod_layout = QHBoxLayout()
        main_layout.addLayout(mod_layout, 3, 1, Qt.AlignCenter)
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
        help = QPushButton("Help", parent)
        help.setToolTip("Show Help")
        help.clicked.connect(self.help)
        bottom_layout.addWidget(help)

        # widget contents/customization:
        self.instructions.setWordWrap(True)
        self.instructions.setText(
            "To customize the Toolbar's Home tab, "
            "use drag and drop to either move buttons and sections around within the Home tab "
            "or to copy them from the Available Buttons.  "
            "Edit the text in the Home tab to change the displayed name.  "
            "Check sections to make them compact.  "
            "All changes are immediately shown in the actual toobar.  "
            "Currently, pulldown menus and mouse modes are unsupported.")

        self.build_home_tab()
        self.home.childDraggedAndDropped.connect(self.update)

        self.other.setColumnCount(1)
        self.other.setDragEnabled(True)
        self.other.setDropIndicatorShown(True)
        self.other.setDragDropMode(QAbstractItemView.DragOnly)
        self.other.setHeaderLabels(["Available Buttons"])
        other_flags = Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsSelectable
        toolbar = self.session.toolbar._toolbar  # internal manager data
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
                tab_item = QTreeWidgetItem(self.other, [tab])
                tab_item.setData(0, ITEM_TYPE_ROLE, TAB_TYPE)
                tab_item.setFlags(Qt.ItemIsEnabled)
                self.other.expandItem(tab_item)
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(tab_item, [section])
                section_item.setData(0, ITEM_TYPE_ROLE, SECTION_TYPE)
                section_item.setFlags(other_flags)
                # Treat all available section as not compact
                # section_item.setCheckState(0, Qt.Checked if compact else Qt.Unchecked)
                self.other.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name}"])
            item.setData(0, ITEM_TYPE_ROLE, BUTTON_TYPE)
            item.setData(0, LINK_ROLE, f"{bundle_info.name}:{name}")
            item.setFlags(other_flags)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)
                item.setIcon(0, icon)
            item.setToolTip(0, description)

    def build_home_tab(self):
        # the following is very similar to code for toolbar layout
        from Qt.QtCore import Qt
        from Qt.QtGui import QIcon
        if self.home.topLevelItemCount() != 0:
            self.home.itemChanged.disconnect()
            self.home.clear()
        self.home.setHeaderLabels(["Home Tab"])
        last_section = None
        section_item = None
        for (section, compact, display_name, icon_path, description, link, bi, name, kw) in _home_layout(self.session, _settings.home_tab):
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(self.home, [section])
                section_item.setData(0, ITEM_TYPE_ROLE, SECTION_TYPE)
                section_item.setFlags(SECTION_FLAGS)
                section_item.setCheckState(0, Qt.Checked if compact else Qt.Unchecked)
                self.home.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name}"])
            item.setData(0, ITEM_TYPE_ROLE, BUTTON_TYPE)
            item.setData(0, LINK_ROLE, link)
            item.setFlags(BUTTON_FLAGS)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)
                item.setIcon(0, icon)
            item.setToolTip(0, description)
        if self.home.topLevelItemCount() != 0:
            self.home.itemChanged.connect(self.update)

    def update(self, *args):
        # check if text of current section item is a duplicate
        if args:
            item, column = args
        else:
            item = None
        if item and item.data(0, ITEM_TYPE_ROLE) == SECTION_TYPE:
            # make sure section name is unique
            from collections import Counter
            section_name = item.text(0)
            parent = item.parent()
            if parent is None:
                parent = self.home.invisibleRootItem()
            current_section_names = []
            for i in range(parent.childCount()):
                item_name = parent.child(i).text(0)
                current_section_names.append(item_name)
            current_sections = Counter(current_section_names)
            if current_sections[section_name] > 1:
                from itertools import chain, count
                for suffix in chain(("",), count(2)):
                    new_name = f"new {section_name}{suffix}"
                    if new_name not in current_sections:
                        item.setText(0, new_name)
                        break
        # propagate user changes to home tab
        from Qt.QtWidgets import QTreeWidgetItemIterator
        home_tab = []
        cur_section = []
        it = QTreeWidgetItemIterator(self.home)
        while it.value():
            item = it.value()
            it += 1
            item_type = item.data(0, ITEM_TYPE_ROLE)
            if item_type == BUTTON_TYPE:
                display_name = item.text(0)
                link = item.data(0, LINK_ROLE)
                # TODO: examine linked item to see if display_name is the same
                name = link.split(sep=':', maxsplit=1)[1]
                if name == display_name:
                    cur_section.append(link)
                else:
                    cur_section.append((link, display_name))
            elif item_type == SECTION_TYPE:
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
        current_sections = set()
        for i in range(self.home.topLevelItemCount()):
            item_name = self.home.topLevelItem(i).text(0)
            current_sections.add(item_name)
        from itertools import chain, count
        for suffix in chain(("",), count(2)):
            new_name = f"new section{suffix}"
            if new_name not in current_sections:
                section_item = QTreeWidgetItem(self.home, [new_name])
                section_item.setData(0, ITEM_TYPE_ROLE, SECTION_TYPE)
                section_item.setFlags(SECTION_FLAGS)
                section_item.setCheckState(0, Qt.Unchecked)
                self.home.expandItem(section_item)
                self.home.scrollToItem(section_item)
                return

    def remove(self):
        # remove selected sections/buttons from home tab
        for si in self.home.selectedItems():
            parent = si.parent()
            if parent:
                parent.removeChild(si)
            else:
                self.home.invisibleRootItem().removeChild(si)
        self.update()

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

    def help(self):
        from chimerax.help_viewer import show_url
        show_url(self.session, self.help_url)

# Adapted QHLine from
# https://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt


class QHLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
