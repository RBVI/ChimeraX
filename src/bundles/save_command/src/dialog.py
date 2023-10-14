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

class MainSaveDialog:
    def __init__(self, settings=None):
        self._settings = settings

    def display(self, session, *, parent=None, format=None, initial_directory=None, initial_file=None):
        if parent is None:
            parent = session.ui.main_window
        from chimerax.ui.open_save import SaveDialog
        dialog = SaveDialog(session, parent, "Save File", installed_only=False)
        self._customize_dialog(session, dialog)
        if format is not None:
            try:
                filter = self._fmt_name2filter[format]
            except KeyError:
                session.logger.warning("Unknown format requested for save dialog: '%s'" % format)
            else:
                dialog.selectNameFilter(filter)
                self._format_selected(session, dialog)
        if initial_directory is not None:
            if initial_directory == '':
                from os import getcwd
                initial_directory = getcwd()
            dialog.setDirectory(initial_directory)
        if initial_file is not None:
            dialog.selectFile(initial_file)
        if not dialog.exec():
            return
        fmt = self._filter2fmt[dialog.selectedNameFilter()]
        save_mgr = session.save_command
        provider_info = save_mgr.provider_info(fmt)
        from chimerax.core.commands import run, SaveFileNameArg, StringArg
        fname = self._add_missing_file_suffix(dialog.selectedFiles()[0], fmt)
        cmd = "save %s" % SaveFileNameArg.unparse(fname)
        if provider_info.bundle_info.installed and self._current_option != self._no_options_label:
            cmd += ' ' + save_mgr.save_args_string_from_widget(fmt, self._current_option)
        if not provider_info.is_default:
            cmd += ' format ' + fmt.nicknames[0]
        run(session, cmd)
        if self._settings:
            self._settings.format_name = fmt.name

    def _add_missing_file_suffix(self, path, fmt):
        import os.path
        ext = os.path.splitext(path)[1]
        if ext not in fmt.suffixes:
            path += fmt.suffixes[0]
        return path

    def _customize_dialog(self, session, dialog):
        options_panel = dialog.custom_area
        saveable_formats = dialog.data_formats
        file_filters = dialog.name_filters
        self._fmt_name2filter = dict(zip([fmt.name for fmt in saveable_formats], file_filters))
        self._filter2fmt = dict(zip(file_filters, saveable_formats))
        file_filters.sort(key=lambda f: f.lower())
        if self._settings:
            try:
                file_filter = self._fmt_name2filter[self._settings.format_name]
            except KeyError:
                pass
            else:
                dialog.selectNameFilter(file_filter)
        from Qt.QtWidgets import QHBoxLayout, QLabel
        self._current_option = self._no_options_label = QLabel(
            "No user-settable options")
        self._options_layout = QHBoxLayout()
        self._options_layout.addWidget(self._no_options_label)
        dialog.custom_area.setLayout(self._options_layout)
        dialog.filterSelected.connect(
            lambda *args, ses=session, dlg=dialog: self._format_selected(ses, dlg))
        self._format_selected(session, dialog)

    def _format_selected(self, session, dialog):
        fmt = self._filter2fmt[dialog.selectedNameFilter()]
        if self._current_option:
            self._current_option.hide()
        from .manager import SaverNotInstalledError
        save_mgr = session.save_command
        try:
            self._current_option = save_mgr.save_args_widget(fmt) or self._no_options_label
        except SaverNotInstalledError:
            from Qt.QtWidgets import QLabel
            from chimerax.core import toolshed
            bi = save_mgr.provider_info(fmt).bundle_info
            self._current_option = QLabel('<a href="%s">Install the %s bundle</a> to save "%s" format files.'
                % (toolshed.get_toolshed().bundle_url(bi.name), bi.short_name, fmt.name))
            from chimerax.core.commands import run
            self._current_option.linkActivated.connect(lambda *args, run=run, name=bi.short_name,
                ses=session, dlg=dialog: (dlg.reject(), run(ses, "toolshed show " + name)))
        self._options_layout.addWidget(self._current_option)
        self._current_option.show()


_settings = None
from chimerax.core.settings import Settings
class SaveDialogSettings(Settings):
    AUTO_SAVE = {
        'format_name': 'ChimeraX session'
    }

def create_menu_entry(session):
    session.ui.main_window.add_menu_entry(["File"], "&Save...",
        lambda *args, ses=session: show_save_file_dialog(ses), tool_tip="Save output file",
            shortcut="Ctrl+S", insertion_point="Set Working Folder...")

_dlg = None
def show_save_file_dialog(session, **kw):
    global _dlg
    if _dlg is None:
        global _settings
        if not _settings:
            _settings = SaveDialogSettings(session, "main save dialog")
        _dlg = MainSaveDialog(settings=_settings)
    _dlg.display(session, **kw)
