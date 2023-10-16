# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Unless you need to add custom widgets to the dialog, you should use Qt.QtWidgets.QFileDialog
# for opening files, since that will have native look and feel.  The OpenDialog below is for
# those situations where you do need to add widgets.
try:
    from Qt.QtWidgets import QFileDialog, QDialog
    from Qt.QtCore import Qt
except ImportError:
    # nogui
    pass
else:
    class OpenDialog(QFileDialog):
        def __init__(self, parent=None, caption='Open File', starting_directory=None,
                     widget_alignment=Qt.AlignCenter, filter=''):
            if starting_directory is None:
                import os
                starting_directory = os.getcwd()
            QFileDialog.__init__(self, parent, caption=caption, directory=starting_directory,
                                 filter=filter)
            self.setFileMode(QFileDialog.AnyFile)
            self.setOption(QFileDialog.DontUseNativeDialog)

            from Qt.QtWidgets import QWidget
            self.custom_area = QWidget()
            layout = self.layout()
            row = layout.rowCount()
            layout.addWidget(self.custom_area, row, 0, 1, -1, widget_alignment)

        def get_path(self):
            if not self.exec():
                return None
            paths = self.selectedFiles()
            if not paths:
                return None
            path = paths[0]
            return path

        def get_paths(self):
            if not self.exec():
                return None
            paths = self.selectedFiles()
            if not paths:
                return None
            return paths

    class OpenFolderDialog(OpenDialog):
        def __init__(self, parent, session):
            OpenDialog.__init__(self, parent=parent, caption='Open Folder')
            from Qt.QtWidgets import QFileDialog
            self.setFileMode(QFileDialog.Directory)

            self._customize_file_dialog(session)

        def _customize_file_dialog(self, session):
            from Qt.QtWidgets import QComboBox, QHBoxLayout, QLabel, QFrame
            options_panel = self.custom_area
            label = QLabel(options_panel)
            label.setText("Format:")
            self._format_selector = selector = QComboBox(options_panel)
            fmt_names = sorted([fmt.synopsis for fmt in session.open_command.open_data_formats
                         if fmt.nicknames and fmt.allow_directory])
            selector.addItems(fmt_names)
            options_layout = QHBoxLayout(options_panel)
            options_layout.addWidget(label)
            options_layout.addWidget(selector)
            options_panel.setLayout(options_layout)
            return options_panel

        def display(self, session):
            from chimerax.core.filehistory import file_history
            fh = file_history(session)
            hfiles = [f for f in fh.files if f.path and f.database is None]
            if hfiles:
                from os.path import dirname
                initial_dir = dirname(hfiles[-1].path)
                self.setDirectory(initial_dir)
            if not self.exec():
                return
            dirs = self.selectedFiles()
            dir = dirs[0] if len(dirs) > 0 else self.directory().path()
            fmt_synopsis = self._format_selector.currentText()
            for fmt in session.open_command.open_data_formats:
                if fmt.synopsis == fmt_synopsis:
                    break
            from chimerax.core.commands import run, FileNameArg
            cmd = 'open %s format %s' % (FileNameArg.unparse(dir), fmt.nicknames[0])
            run(session, cmd)

    class FetchDialog(QDialog):
        help = None

        def __init__(self, session, *, debug=False):
            from Qt.QtWidgets import QAbstractItemView, QVBoxLayout, QLabel, QDialogButtonBox as qbbox
            from Qt.QtWidgets import QLineEdit, QHBoxLayout, QWidget, QCheckBox
            from chimerax.ui.widgets import ItemTable
            from chimerax.ui import shrink_font
            super().__init__()
            self.debug = debug
            self.session = session
            self.setWindowTitle("Fetch By ID")
            self.handler = session.open_command.triggers.add_handler("open command changed",
                self.set_table_data)
            layout = QVBoxLayout()
            layout.setSpacing(2)
            self.setLayout(layout)
            self.choose_in_table = QLabel("Choose database:")
            shrink_font(self.choose_in_table, 0.85)
            layout.addWidget(self.choose_in_table)
            self.table = ItemTable(allow_user_sorting=False, auto_multiline_headers=False)
            self.table.add_column("Database", "database")
            if self.debug:
                self.table.add_column("format", "format")
                self.table.add_column("nickname", "nickname")
            self.table.add_column("Example IDs", "example_ids")
            self.set_table_data()
            self.table.launch(select_mode=QAbstractItemView.SelectionMode.SingleSelection)
            self.table.resizeRowsToContents()
            self.table.selection_changed.connect(self.update_entry_area)
            layout.addWidget(self.table, alignment=Qt.AlignCenter)
            self.entry_area = QWidget()
            layout.addWidget(self.entry_area)
            entry_layout = QHBoxLayout()
            entry_layout.setSpacing(2)
            self.entry_label = QLabel()
            entry_layout.addWidget(self.entry_label)
            self.id_entry = QLineEdit()
            entry_layout.addWidget(self.id_entry)
            self.entry_area.setLayout(entry_layout)
            self.update_entry_area()
            self.ignore_cache = QCheckBox("Ignore cached fetches")
            shrink_font(self.ignore_cache)
            layout.addWidget(self.ignore_cache, alignment=Qt.AlignCenter)

            bbox = qbbox(qbbox.Close | qbbox.Help)
            bbox.addButton("Fetch", qbbox.AcceptRole)
            bbox.accepted.connect(self.fetch)
            bbox.rejected.connect(self.reject)
            if self.help:
                from chimerax.core.commands import run
                bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
            else:
                bbox.button(qbbox.Help).setEnabled(False)
            layout.addWidget(bbox)

            from chimerax.core.settings import Settings
            class FetchPanelSettings(Settings):
                AUTO_SAVE = {
                    'prev_db': None
                }
            self.settings = FetchPanelSettings(session, "Fetch By ID")
            if self.settings.prev_db:
                self.set_database(self.settings.prev_db)

        def fetch(self):
            from chimerax.core.errors import UserError
            table_sel = self.table.selected
            if not table_sel:
                raise UserError("Select a database from the database table")
            db_info = table_sel[0]
            fetch_id = self.id_entry.text()
            if not fetch_id:
                raise UserError("Enter a %s ID in the entry field" % db_info.database)
            data_format = self.session.data_formats[db_info.format]
            from chimerax.core.commands import run, StringArg
            cmd = "open %s from %s format %s" % (
                StringArg.unparse(fetch_id),
                StringArg.unparse(db_info.nickname),
                StringArg.unparse(data_format.nicknames[0]),
            )
            if self.ignore_cache.isChecked():
                cmd += " ignoreCache true"
            self.hide()
            self.settings.prev_db = db_info.database
            run(self.session, cmd)

        def set_database(self, db_name):
            for db_info in self.table.data:
                if db_info.database == db_name:
                    self.table.selected = [db_info]
                    break

        def set_table_data(self, *args):
            class FetchDBInfo:
                def __init__(self, database, format, example_ids, nickname):
                    self._database = database
                    self.format = format
                    self.example_ids = example_ids
                    self.nickname = nickname

                def __eq__(self, other):
                    return self.database == other.database and self.format == other.format

                def __lt__(self, other):
                    if self.database is None or other.database is None:
                        return self.nickname < other.nickname
                    return self.database.lower() < other.database.lower()

                def __hash__(self):
                    return id(self)

                @property
                def database(self):
                    if self._database is None:
                        return self._database
                    return '\n'.join(self._database.split(' ; '))

            table_data = []
            database_names = self.session.open_command.database_names
            for db_name in database_names:
                db_info = self.session.open_command.database_info(db_name)
                for format, fetcher_info in db_info.items():
                    database = fetcher_info.synopsis
                    if not database and not self.debug:
                        continue
                    if fetcher_info.example_ids:
                        example_ids = '\n'.join(fetcher_info.example_ids)
                    else:
                        example_ids = None
                    data_item = FetchDBInfo(database, format, example_ids, db_name)
                    table_data.append(data_item)
            table_data.sort()
            self.table.data = table_data

        def update_entry_area(self, *args):
            sel = self.table.selected
            if sel:
                db_info = sel[0]
                if db_info.database:
                    db = db_info.database.split('\n')[0]
                else:
                    db = db_info.nickname
                self.entry_label.setText("Enter %s ID:" % db)
                self.id_entry.clear()
            self.entry_area.setHidden(not sel)

def create_menu_entry(session):
    # only folder format right now is DICOM, so use hard coded menu entry for now
    session.ui.main_window.add_menu_entry(["File"], "Open DICOM Folder...",
        lambda *args, ses=session: show_open_folder_dialog(ses), tool_tip="Open folder data",
            insertion_point=False)
    session.ui.main_window.add_menu_entry(["File"], "&Fetch By ID...",
        lambda *args, ses=session: show_fetch_by_id_dialog(ses), tool_tip="Fetch files from web",
            shortcut="Ctrl+F", insertion_point=False)
    session.ui.main_window.add_menu_entry(["File"], "&Open...",
        lambda *args, ses=session: show_open_file_dialog(ses), tool_tip="Open input file",
            shortcut="Ctrl+O", insertion_point=False)

_use_native_open_file_dialog = True
def set_use_native_open_file_dialog(use):
    global _use_native_open_file_dialog
    _use_native_open_file_dialog = use

def make_qt_name_filters(session, *, no_filter="All files (*)"):
    openable_formats = [fmt for fmt in session.open_command.open_data_formats if fmt.suffixes]
    openable_formats.sort(key=lambda fmt: fmt.synopsis.casefold())
    file_filters = ["%s (%s)" % (fmt.synopsis, "*" + " *".join(fmt.suffixes))
        for fmt in openable_formats]
    if no_filter is not None:
        file_filters = [no_filter] + file_filters
    return file_filters, openable_formats, no_filter

_fetch_by_id_dialog = None
def show_fetch_by_id_dialog(session, database_name=None, *, debug=False):
    global _fetch_by_id_dialog
    if _fetch_by_id_dialog is None:
        _fetch_by_id_dialog = FetchDialog(session, debug=debug)

    if database_name is not None:
        _fetch_by_id_dialog.set_database(database_name)
    _fetch_by_id_dialog.show()
    _fetch_by_id_dialog.raise_()

def show_open_file_dialog(session, initial_directory=None, format_name=None):
    if initial_directory is None:
        initial_directory = ''
    file_filters, openable_formats, no_filter = make_qt_name_filters(session)
    fmt_name2filter = dict(zip([fmt.name for fmt in openable_formats], file_filters[1:]))
    filter2fmt = dict(zip(file_filters[1:], openable_formats))
    filter2fmt[no_filter] = None
    from Qt.QtWidgets import QFileDialog
    qt_filter = ";;".join(file_filters)
    if _use_native_open_file_dialog:
        from Qt.QtWidgets import QFileDialog
        paths, file_filter = QFileDialog.getOpenFileNames(filter=qt_filter,
                                                       directory=initial_directory)
        from sys import platform
        if platform == 'win32':
            # On Windows 10 the native open dialog puts "fatal errors" into the
            # faulthandler crash monitoring file that are in fact not fatal.
            # These make Windows crash reports hard to understand.  So clear them.
            from chimerax.bug_reporter import crash_report
            crash_report.clear_fault_handler_file(session)
    else:
        dlg = OpenDialog(parent=session.ui.main_window, starting_directory=initial_directory,
                       filter=qt_filter)
        dlg.setNameFilters(file_filters)
        paths = dlg.get_paths()
        file_filter = dlg.selectedNameFilter()

    if not paths:
        return

    # Linux doesn't return a valid file_filter if none is chosen
    if not file_filter:
        data_format = None
    else:
        data_format = filter2fmt[file_filter]

    def _qt_safe(session=session, paths=paths, data_format=data_format):
        from chimerax.core.commands import run, FileNameArg, StringArg
        run(session, "open " + " ".join([FileNameArg.unparse(p) for p in paths]) + (""
            if data_format is None else " format " + StringArg.unparse(data_format.nicknames[0])))
    # Opening the model directly adversely affects Qt interfaces that show
    # as a result.  In particular, Multalign Viewer no longer gets hover
    # events correctly, nor tool tips.
    #
    # Using session.ui.thread_safe() doesn't help either(!)
    from Qt.QtCore import QTimer
    QTimer.singleShot(0, _qt_safe)

_folder_dlg = None
def show_open_folder_dialog(session):
    global _folder_dlg
    if _folder_dlg is None:
        _folder_dlg = OpenFolderDialog(session.ui.main_window, session)
    _folder_dlg.display(session)
