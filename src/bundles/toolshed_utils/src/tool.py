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
    SESSION_SAVE = False
    help = "help:user/tools/updates.html"

    NAME_COLUMN = 0
    CURRENT_VERSION_COLUMN = 1
    NEW_VERSION_COLUMN = 2
    CATEGORY_COLUMN = 3
    NUM_COLUMNS = CATEGORY_COLUMN + 1

    def __init__(self, session, tool_name, dialog_type=None):
        if dialog_type is None:
            dialog_type = DialogType.UPDATES_ONLY
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from Qt.QtCore import Qt
        from Qt.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QPushButton, QLabel, QComboBox
        layout = QVBoxLayout()
        parent.setLayout(layout)
        label = QLabel(
            "<p>Select the individual bundles you want to install by checking the box."
            "  Then click on the <b>Install</b> button to install them."
            "  The default bundle version is the newest one, but you can select an older version."
            "<p>When only showing updates, check <b>All</b> to select all of them."
        )
        label.setWordWrap(True)
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
        from chimerax.ui.core_settings_ui import UpdateIntervalOption
        from chimerax.core.core_settings import settings as core_settings
        callback = None  # TODO
        uio = UpdateIntervalOption(
                "Toolshed update interval", core_settings.toolshed_update_interval, callback,
                attr_name='toolshed_update_interval', settings=core_settings, auto_set_attr=True)
        label = QLabel(uio.name + ':')
        label.setToolTip('How frequently to check toolshed for new updates<br>')
        choice_layout.addWidget(label)
        choice_layout.addWidget(uio.widget)
        self.all_items = None

        class SizedTreeWidget(QTreeWidget):
            def sizeHint(self):
                from Qt.QtCore import QSize
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
        button = QPushButton("Help")
        button.clicked.connect(self.help_button)
        buttons_layout.addWidget(button)
        buttons_layout.addStretch()
        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self.install)
        self.install_button.setEnabled(False)
        buttons_layout.addWidget(self.install_button)
        self.updates.itemClicked.connect(self.update_install_button)
        button = QPushButton("Cancel")
        button.clicked.connect(self.cancel)
        buttons_layout.addWidget(button)

        self._fill_updates()
        self.tool_window.fill_context_menu = self.fill_context_menu
        self.tool_window.manage(placement=None)

    def help_button(self):
        from chimerax.help_viewer import show_url
        show_url(self.session, self.help, new_tab=True)

    def cancel(self):
        self.delete()

    def update_install_button(self, *args):
        from Qt.QtCore import Qt
        all_items = self.all_items
        for i in range(all_items.childCount()):
            item = all_items.child(i)
            if item.checkState(self.NAME_COLUMN) == Qt.Checked:
                self.install_button.setEnabled(True)
                return
        self.install_button.setEnabled(False)

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction
        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(lambda arg: self.show_settings())
        menu.addAction(settings_action)

    def show_settings(self):
        self.session.ui.main_window.show_settings('Toolshed')

    def _fill_updates(self):
        from Qt.QtCore import Qt
        from Qt.QtWidgets import QTreeWidgetItem, QComboBox
        from packaging.version import Version
        # TODO: make _compatible a non-private API
        from chimerax.help_viewer.tool import _compatible as compatible
        session = self.session
        toolshed = session.toolshed
        self.actions = []
        info = toolshed.bundle_info(session.logger, installed=False, available=True)
        dialog_type = self.choice.currentData()
        new_bundles = {}
        last_bundle_name = None
        installed_version = ""
        for available in info:
            release_file = getattr(available, 'release_file', None)
            if release_file is not None and not compatible(release_file):
                continue
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

        self.all_items = None
        self.updates.clear()
        if not new_bundles:
            return
        if dialog_type != DialogType.UPDATES_ONLY:
            self.all_items = all_items = self.updates.invisibleRootItem()
        else:
            self.all_items = all_items = QTreeWidgetItem()
            all_items.setText(0, "All")
            all_items.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            all_items.setCheckState(self.NAME_COLUMN, Qt.Unchecked)
            self.updates.addTopLevelItem(all_items)
        self.updates.expandItem(all_items)
        flags = Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemNeverHasChildren
        for bundle_name in new_bundles:
            new_versions, installed_version, synopsis, category = new_bundles[bundle_name]
            item = QTreeWidgetItem(all_items)
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
        from Qt.QtCore import Qt
        toolshed = self.session.toolshed
        logger = self.session.logger
        all_items = self.all_items
        updating = []
        for i in range(all_items.childCount()):
            item = all_items.child(i)
            if item.checkState(self.NAME_COLUMN) == Qt.Unchecked:
                continue
            bundle_name = item.data(self.NAME_COLUMN, Qt.UserRole)
            w = self.updates.itemWidget(item, self.NEW_VERSION_COLUMN)
            version = w.currentText()
            bi = toolshed.find_bundle(bundle_name, logger, version=version, installed=False)
            updating.append(bi)
        from . import _install_bundle
        _install_bundle(toolshed, updating, logger, reinstall=True, session=self.session)
        self.cancel()
