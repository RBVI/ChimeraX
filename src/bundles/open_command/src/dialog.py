# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Unless you need to add custom widgets to the dialog, you should use Qt.QtWidgets.QFileDialog
# for opening files, since that will have native look and feel.  The OpenDialog below is for
# those situations where you do need to add widgets.
try:
    from Qt.QtWidgets import QFileDialog
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
            fmt_names = [fmt.synopsis for fmt in session.open_command.open_data_formats
                         if fmt.nicknames and fmt.allow_directory]
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

def create_menu_entry(session):
    # only folder format right now is DICOM, so use hard coded menu entry for now
    session.ui.main_window.add_menu_entry(["File"], "Open DICOM Folder...",
        lambda *args, ses=session: show_open_folder_dialog(ses), tool_tip="Open folder data",
            insertion_point=False)
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
