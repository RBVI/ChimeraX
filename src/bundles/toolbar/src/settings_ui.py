# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2019 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""Toolbar customization widgets, loaded when Settings is opened."""

from Qt.QtWidgets import QFrame, QTreeWidget, QTreeWidgetItem
from Qt.QtCore import Qt, Signal
from . import tool as _toolbar

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
            new_section = self.itemAt(event.position().toPoint())
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
            new_button = self.itemAt(event.position().toPoint())
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
        for (tab, section, compact, display_name, icon_info, description, bundle_info, name, kw) in _toolbar._other_layout(self.session, toolbar, hide_hidden=False):
            theme_icon, icon_path, dark_icon_path = icon_info
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
            if theme_icon is not None:
                icon = QIcon.fromTheme(theme_icon)
            elif dark_icon_path is not None and self.session.ui.dark_mode():
                icon = QIcon(dark_icon_path)
            elif icon_path is not None:
                icon = QIcon(icon_path)
            else:
                icon = None
            if icon is not None:
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
        for (section, compact, display_name, icon_info, description, link, bi, name, kw) in _toolbar._home_layout(self.session, _toolbar._settings.home_tab):
            theme_icon, icon_path, dark_icon_path = icon_info
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
            if theme_icon is not None:
                icon = QIcon.fromTheme(theme_icon)
            elif dark_icon_path is not None and self.session.ui.dark_mode():
                icon = QIcon(dark_icon_path)
            elif icon_path is not None:
                icon = QIcon(icon_path)
            else:
                icon = None
            if icon is not None:
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
                if item.checkState(0) != Qt.Unchecked:
                    home_tab.append(((name, True), cur_section))
                else:
                    home_tab.append((name, cur_section))
        _toolbar._settings.home_tab = home_tab
        tb = _toolbar.get_toolbar_singleton(self.session, create=False)
        if tb:
            tb.build_home_tab()

    def update_from_settings(self):
        self.build_home_tab()
        tb = _toolbar.get_toolbar_singleton(self.session, create=False)
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
        _toolbar._settings.save()

    def reset(self):
        # reset current configuration in original defaults
        _toolbar._settings.reset()
        self.update_from_settings()

    def restore(self):
        # restore current configuration from saved preferences
        _toolbar._settings.restore()
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
