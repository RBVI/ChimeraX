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

from chimerax.core.tools import ToolInstance


def show(session):
    return UpdateTool(session, "Update Bundles")


class UpdateTool(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    help = "help:user/tools/updatetool.html"

    NAME_COLUMN = 0
    CURRENT_VERSION_COLUMN = 1
    NEW_VERSION_COLUMN = 2
    CATEGORY_COLUMN = 3
    NUM_COLUMNS = CATEGORY_COLUMN + 1

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QPushButton, QLabel
        layout = QVBoxLayout()
        parent.setLayout(layout)
        label = QLabel(
            "<h1><b>Available Updates:</b></h1>"
            "<p>Check individual bundles you want to update, or click <b>All</b> for all of them."
            "<br>Then click on the <b>Update</b> button to do the update.")
        layout.addWidget(label)
        # self.updates = QTreeWidget(parent)

        class SizedTreeWidget(QTreeWidget):
            def sizeHint(self):
                from PyQt5.QtCore import QSize
                width = self.header().length()
                return QSize(width, 200)
        self.updates = SizedTreeWidget()
        layout.addWidget(self.updates)
        self.updates.setHeaderLabels(["Bundle\nName", "Current\nVersion", "New\nVersion", "Category"])
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
        button = QPushButton("Update")
        button.clicked.connect(self.update)
        buttons_layout.addWidget(button)
        button = QPushButton("Cancel")
        button.clicked.connect(self.cancel)
        buttons_layout.addWidget(button)

        self._fill_updates()
        self.tool_window.manage(placement=None)

    def cancel(self):
        self.session.ui.main_window.close_request(self.tool_window)

    def _fill_updates(self, *, always_rebuild=False):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidgetItem, QComboBox
        from packaging.version import Version
        session = self.session
        toolshed = session.toolshed
        self.actions = []
        info = toolshed.bundle_info(session.logger, installed=False, available=True)
        new_bundles = {}
        installed_bi = None
        for available in info:
            if installed_bi is None or available.name != installed_bi.name:
                installed_bi = toolshed.find_bundle(available.name, session.logger)
                if installed_bi is None:
                    continue
                installed_version = Version(installed_bi.version)
            new_version = Version(available.version)
            if True or new_version > installed_version:  # DEBUG
                new_bundles.setdefault(installed_bi, []).append(new_version)
        self.all_item = all_item = QTreeWidgetItem()
        all_item.setText(0, "All")
        all_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        all_item.setCheckState(self.NAME_COLUMN, Qt.Unchecked)
        self.updates.addTopLevelItem(all_item)
        self.updates.expandItem(all_item)
        flags = Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemNeverHasChildren
        # for row, bi in enumerate(sorted(new_bundles, key=lambda x: x.name.casefold())):
        for bi in new_bundles:
            new_versions = new_bundles[bi]
            item = QTreeWidgetItem(all_item)
            # Name column
            name = bi.name
            item.setData(self.NAME_COLUMN, Qt.UserRole, name)
            if name.startswith("ChimeraX-"):
                name = name[len("ChimeraX-"):]
            item.setText(self.NAME_COLUMN, name)
            item.setFlags(flags)
            item.setToolTip(self.NAME_COLUMN, bi.synopsis)
            item.setCheckState(self.NAME_COLUMN, Qt.Unchecked)
            # Current version column
            item.setText(self.CURRENT_VERSION_COLUMN, bi.version)
            item.setTextAlignment(self.CURRENT_VERSION_COLUMN, Qt.AlignCenter)
            # New version column
            b = QComboBox()
            b.addItems(str(v) for v in sorted(new_versions, reverse=True))
            # TODO: set background color to same as other columns
            self.updates.setItemWidget(item, self.NEW_VERSION_COLUMN, b)

            # Category column
            item.setText(self.CATEGORY_COLUMN, bi.categories[0])

        self.updates.sortItems(self.NAME_COLUMN, Qt.AscendingOrder)
        for column in range(self.NUM_COLUMNS):
            self.updates.resizeColumnToContents(column)

    def update(self):
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
