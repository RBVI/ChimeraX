# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2020 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from enum import Enum
from chimerax.core.tools import ToolInstance


class DialogType(Enum):
    ALL_AVAILABLE = 0
    INSTALLED_ONLY = 1
    UPDATES_ONLY = 2


def show(session, dialog_type=None):
    if not session.ui.is_gui:
        return
    return UpdateTool(session, "Update Bundles", dialog_type)


class UpdateTool(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    help = "help:user/tools/updatetool.html"

    NAME_COLUMN = 0
    CURRENT_VERSION_COLUMN = 1
    NEW_VERSION_COLUMN = 2
    CATEGORY_COLUMN = 3
    NUM_COLUMNS = CATEGORY_COLUMN + 1

    def __init__(self, session, tool_name, dialog_type=None):
        if dialog_type is None:
            dialog_type = DialogType.ALL_AVAILABLE
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QPushButton, QLabel, QComboBox, QSizePolicy
        layout = QVBoxLayout()
        parent.setLayout(layout)
        label = QLabel(
            "<p>Check individual bundles you want to update, or click <b>All</b> for all of them."
            "<br>Then click on the <b>Install</b> button to install them.")
        layout.addWidget(label)
        choice_layout = QHBoxLayout()
        layout.addLayout(choice_layout)
        label = QLabel("<b>Show:</b>")
        choice_layout.addWidget(label)
        self.choice = QComboBox()
        choice_layout.addWidget(self.choice)
        for dt in DialogType:
            self.choice.addItem(dt.name.replace('_', ' ').title(), dt)
        self.choice.setCurrentIndex(self.choice.findData(dialog_type))
        self.choice.currentIndexChanged.connect(self.new_choice)
        choice_layout.addStretch()

        class SizedTreeWidget(QTreeWidget):
            def sizeHint(self):
                from PyQt5.QtCore import QSize
                width = self.header().length()
                return QSize(width, 200)
        self.updates = SizedTreeWidget()
        # self.updates = QTreeWidget(parent)
        layout.addWidget(self.updates)
        self.updates.setHeaderLabels(["Bundle\nName", "Current\nVersion", "Available\nVersion(s)", "Category"])
        self.updates.setSortingEnabled(True)
        hi = self.updates.headerItem()
        hi.setTextAlignment(self.NAME_COLUMN, Qt.AlignCenter)
        hi.setTextAlignment(self.CURRENT_VERSION_COLUMN, Qt.AlignCenter)
        hi.setTextAlignment(self.NEW_VERSION_COLUMN, Qt.AlignCenter)

        self.updates.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.updates.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.updates.setEditTriggers(QAbstractItemView.NoEditTriggers)
        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout)
        buttons_layout.addStretch()
        button = QPushButton("Install")
        button.clicked.connect(self.install)
        buttons_layout.addWidget(button)
        button = QPushButton("Cancel")
        button.clicked.connect(self.cancel)
        buttons_layout.addWidget(button)

        self._fill_updates()
        self.tool_window.manage(placement=None)

    def cancel(self):
        self.session.ui.main_window.close_request(self.tool_window)

    def fill_context_menu(self, menu, x, y):
        from PyQt5.QtWidgets import QAction
        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(lambda arg: self.show_settings())
        menu.addAction(settings_action)

    def show_settings(self):
        pass
        # TODO: show toolshed settings

    def _fill_updates(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidgetItem, QComboBox
        from packaging.version import Version
        session = self.session
        toolshed = session.toolshed
        self.actions = []
        info = toolshed.bundle_info(session.logger, installed=False, available=True)
        dialog_type = self.choice.currentData()
        new_bundles = {}
        last_bundle_name = None
        installed_version = ""
        for available in info:
            if last_bundle_name is None or available.name != last_bundle_name:
                last_bundle_name = available.name
                installed_bi = toolshed.find_bundle(last_bundle_name, session.logger)
                if installed_bi is not None:
                    installed_version = Version(installed_bi.version)
                else:
                    installed_version = ""
                    if dialog_type != DialogType.ALL_AVAILABLE:
                        continue
            elif not installed_version and dialog_type != DialogType.ALL_AVAILABLE:
                continue
            new_version = Version(available.version)
            if dialog_type == DialogType.UPDATES_ONLY:
                if new_version <= installed_version:
                    continue
            data = new_bundles.setdefault(
                last_bundle_name,
                ([], installed_version, available.synopsis, available.categories[0]))
            data[0].append(new_version)

        self.updates.clear()
        self.all_item = all_item = QTreeWidgetItem()
        all_item.setText(0, "All")
        all_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        all_item.setCheckState(self.NAME_COLUMN, Qt.Unchecked)
        self.updates.addTopLevelItem(all_item)
        self.updates.expandItem(all_item)
        flags = Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemNeverHasChildren
        for bundle_name in new_bundles:
            new_versions, installed_version, synopsis, category = new_bundles[bundle_name]
            item = QTreeWidgetItem(all_item)
            # Name column
            item.setData(self.NAME_COLUMN, Qt.UserRole, bundle_name)
            if False:
                # TODO: need delagate to show Qt Rich Text
                item.setText(self.NAME_COLUMN, toolshed.bundle_link(bundle_name))
            else:
                if bundle_name.startswith("ChimeraX-"):
                    bundle_name = bundle_name[len("ChimeraX-"):]
                item.setText(self.NAME_COLUMN, bundle_name)
            item.setFlags(flags)
            item.setToolTip(self.NAME_COLUMN, synopsis)
            item.setCheckState(self.NAME_COLUMN, Qt.Unchecked)
            # Current version column
            item.setText(self.CURRENT_VERSION_COLUMN, str(installed_version))
            item.setTextAlignment(self.CURRENT_VERSION_COLUMN, Qt.AlignCenter)
            # New version column
            b = QComboBox()
            b.addItems(str(v) for v in sorted(new_versions, reverse=True))
            # TODO: set background color to same as other columns
            self.updates.setItemWidget(item, self.NEW_VERSION_COLUMN, b)

            # Category column
            item.setText(self.CATEGORY_COLUMN, category)

        self.updates.sortItems(self.NAME_COLUMN, Qt.AscendingOrder)
        for column in range(self.NUM_COLUMNS):
            self.updates.resizeColumnToContents(column)

    def new_choice(self):
        self._fill_updates()

    def install(self):
        from PyQt5.QtCore import Qt
        toolshed = self.session.toolshed
        logger = self.session.logger
        all_item = self.all_item
        updating = []
        for i in range(all_item.childCount()):
            item = all_item.child(i)
            if item.checkState(self.NAME_COLUMN) == Qt.Unchecked:
                continue
            bundle_name = item.data(self.NAME_COLUMN, Qt.UserRole)
            w = self.updates.itemWidget(item, self.NEW_VERSION_COLUMN)
            version = w.currentText()
            bi = toolshed.find_bundle(bundle_name, logger, version=version, installed=False)
            updating.append((bi, bi.installed))
        print("UPDATING:", updating)
        self.cancel()
