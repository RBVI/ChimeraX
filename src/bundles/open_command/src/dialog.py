# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Unless you need to add custom widgets to the dialog, you should use Qt.QtWidgets.QFileDialog
# for opening files, since that will have native look and feel.  OpenDialog is for
# those situations where you do need to add widgets.
import sys
if sys.platform == 'win32':
    from chimerax.core.utils import no_garbage_collection as gc_context
else:
    from contextlib import nullcontext as gc_context

_DIALOG_CLASSES = ('OpenDialog', 'OpenFolderDialog', 'FetchDialog')

def __getattr__(name):
    if name in _DIALOG_CLASSES:
        from . import _dialog_gui
        value = getattr(_dialog_gui, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(_DIALOG_CLASSES))

def create_menu_entry(session):
    def folder_type_name(fmt):
        nick = fmt.nicknames[0]
        if nick.upper() in fmt.name:
            return nick.upper()
        return nick.capitalize()
    folder_type_names = []
    for fmt in session.open_command.open_data_formats:
        if fmt.allow_directory and session.open_command.opener_info(fmt) is not None:
            folder_type_names.append(folder_type_name(fmt))
    if folder_type_names:
        if len(folder_type_names) < 4:
            folder_adjective = "/".join(folder_type_names)
        else:
            folder_adjective = "Data"
    else:
        folder_adjective = None
    if folder_adjective is not None:
        session.ui.main_window.add_menu_entry(["File"], "Open %s Folder..." % folder_adjective,
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

def make_qt_name_filters(session, *, no_filter="All files (*)", format_names=None):
    openable_formats = [fmt for fmt in session.open_command.open_data_formats if fmt.suffixes]
    if format_names is not None:
        openable_formats = [fmt for fmt in openable_formats if fmt.name in format_names]
    openable_formats.sort(key=lambda fmt: fmt.synopsis.casefold())
    file_filters = ["%s (%s)" % (fmt.synopsis, "*" + " *".join(fmt.suffixes))
        for fmt in openable_formats]
    if no_filter is not None and format_names is None:
        file_filters = [no_filter] + file_filters
    return file_filters, openable_formats, no_filter

_fetch_by_id_dialog = None
def show_fetch_by_id_dialog(session, database_name=None, *, debug=False):
    global _fetch_by_id_dialog
    if _fetch_by_id_dialog is None:
        from .dialog import FetchDialog
        _fetch_by_id_dialog = FetchDialog(session, debug=debug)

    if database_name is not None:
        _fetch_by_id_dialog.set_database(database_name)
    _fetch_by_id_dialog.show()
    _fetch_by_id_dialog.raise_()

def show_open_file_dialog(session, initial_directory=None, format_names=None, *, caption=None):
    if initial_directory is None:
        initial_directory = ''
    file_filters, openable_formats, no_filter = make_qt_name_filters(session, format_names=format_names)
    fmt_name2filter = dict(zip([fmt.name for fmt in openable_formats], file_filters[1:]))
    if format_names is None:
        filter2fmt = dict(zip(file_filters[1:], openable_formats))
        filter2fmt[no_filter] = None
    else:
        filter2fmt = dict(zip(file_filters, openable_formats))
    from Qt.QtWidgets import QFileDialog
    qt_filter = ";;".join(file_filters)
    # native Mac open dialogs (and possibly others) don't show captions...
    if _use_native_open_file_dialog and caption is None:
        from Qt.QtWidgets import QFileDialog
        with gc_context():
            from Qt import using_pyqt6
            if using_pyqt6:
                paths, file_filter = QFileDialog.getOpenFileNames(filter=qt_filter,
                                                       directory=initial_directory)
            else:
                paths, file_filter = QFileDialog.getOpenFileNames(filter=qt_filter,
                                                       dir=initial_directory)

        from sys import platform
        if platform == 'win32':
            # On Windows 10 the native open dialog puts "fatal errors" into the
            # faulthandler crash monitoring file that are in fact not fatal.
            # These make Windows crash reports hard to understand.  So clear them.
            from chimerax.bug_reporter import crash_report
            crash_report.clear_fault_handler_file(session)
    else:
        from .dialog import OpenDialog
        dlg = OpenDialog(parent=session.ui.main_window, starting_directory=initial_directory,
                       filter=qt_filter, caption=caption)
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
        from .dialog import OpenFolderDialog
        _folder_dlg = OpenFolderDialog(session.ui.main_window, session)
    _folder_dlg.display(session)


__all__ = [name for name in globals() if not name.startswith("_")] + list(_DIALOG_CLASSES)
